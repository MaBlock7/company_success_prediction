from dataclasses import dataclass, asdict
from jarowinkler import jarowinkler_similarity
import pandas as pd
from tqdm import tqdm


pd.set_option('future.no_silent_downcasting', True)


@dataclass
class Person:
    idx: int
    fid: int
    fn: str
    ln: str
    ht: set[str]  # hometowns
    nat: set[str]  # nationalities
    por: set[str]  # places of residence
    gen: str  # gender
    heuristic: int = None
    fn_tokens: set[str] = None  # tokenized first names
    ln_tokens: set[str] = None  # tokenized last names
    name_token_set: set[str] = None  # combined set of all name tokens
    hometown_overlap: set[str] = None  # at least one hometown matches
    nationality_overlap: set[str] = None  # at least one nationality matches
    residence_overlap: set[str] = None  # at least one place of residence matches
    location_match_flag: bool = False  # wether to use the hometown (Swiss) or nationality + place of residence (foreigners) match


class PersonClustering:
    """
    Class for clustering individuals based on names, hometowns, nationalities, and residences across and within companies using various heuristics.

    Attributes:
        df (pd.DataFrame): DataFrame containing raw person data for clustering.
        heuristics_within (list[callable]): List of heuristic methods applied within companies.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = self._prepare_dataframe(df)
        self.df['fid'] = None
        self.df.loc[self.df.founders, 'fid'] = range(1, len(self.df[self.df.founders]) + 1)
        self.heuristics_within = [
            self._heuristic_1, self._heuristic_2,
            self._heuristic_3, self._heuristic_4,
            self._heuristic_5, self._heuristic_6,
            self._heuristic_7, self._heuristic_8
        ]

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame):
        """
        Fills missing person data based on chronological order within groups.
        """
        return (df
                .sort_values('shab_date')
                .groupby(['ehraid', 'first_name_norm', 'last_name_norm'], group_keys=False)[df.columns]
                .apply(lambda group: group.ffill().bfill().infer_objects(copy=False)))

    @staticmethod
    def _overlap(set_i: set, set_j: set) -> float:
        """
        Calculates overlap ratio between two sets.
        """
        intersection = set_i.intersection(set_j)
        minlen = min(len(set_i), len(set_j))
        return len(intersection) / minlen if minlen else 0.0

    @staticmethod
    def _jaro_winkler(name_i: str, name_j: str) -> float:
        """
        Computes Jaro-Winkler similarity between two names.
        """
        return jarowinkler_similarity(name_i, name_j)

    def _heuristic_1(self, founder: Person, person: Person):
        """
        Heuristic 1: Perfect Match.
        """
        if founder.fn == person.fn and founder.ln == person.ln and person.location_match_flag:
            person.heuristic = 1
            return True
        return False

    def _heuristic_2(self, founder: Person, person: Person):
        """
        Heuristic 2: Wrong name order.
        """
        if founder.name_token_set == person.name_token_set and person.location_match_flag:
            person.heuristic = 2
            return True
        return False

    def _heuristic_3(self, founder: Person, person: Person):
        """
        Heuristic 3: Marriage/Divorce scenario (last names partially match).
        """
        if founder.fn_tokens == person.fn_tokens and person.location_match_flag and self._overlap(founder.ln_tokens, person.ln_tokens) == 1:
            person.heuristic = 3
            return True
        return False

    def _heuristic_4(self, founder: Person, person: Person):
        """
        Heuristic 4: Marriage/Divorce scenario (last name changed).
        """
        if founder.fn_tokens == person.fn_tokens and person.location_match_flag:
            person.heuristic = 4
            return True
        return False

    def _heuristic_5(self, founder: Person, person: Person):
        """
        Heuristic 5: Naturalization scenario.
        """
        if founder.name_token_set == person.name_token_set and not bool(person.nationality_overlap):
            person.heuristic = 5
            return True
        return False

    def _heuristic_6(self, founder: Person, person: Person):
        """
        Heuristic 6: Middle Names scenario.
        """
        if founder.ln_tokens == person.ln_tokens and self._overlap(founder.fn_tokens, person.fn_tokens) == 1 and person.location_match_flag:
            person.heuristic = 6
            return True
        return False

    def _heuristic_7(self, founder: Person, person: Person):
        """
        Heuristic 7: Misspelled Names scenario.
        """
        if person.location_match_flag and self._jaro_winkler(founder.fn, person.fn) > 0.8 and self._jaro_winkler(founder.ln, person.ln) > 0.8:
            person.heuristic = 7
            return True
        return False

    def _heuristic_8(self, founder: Person, person: Person):
        """
        Heuristic 8: Missing Values scenario.
        """
        if (not founder.ht or not person.ht) or (not founder.nat or not person.nat):
            if founder.name_token_set == person.name_token_set:
                person.heuristic = 8
                return True
        return False

    def _initialize_clusters(self, company_df: pd.DataFrame, founders: bool = False) -> list[Person]:
        """
        Initializes clusters from company data, distinguishing founders if specified.
        """
        subset_df = company_df[company_df.founders] if founders else company_df[~company_df.founders]
        return [
            Person(
                idx=r_idx,
                fid=row.fid,
                fn=row.first_name_norm,
                ln=row.last_name_norm,
                ht=set(
                    row[[f'hometown_{i}_bfs_gmde_code_latest' for i in range(1, 6)]]
                    .dropna()
                    .astype(str)
                ),
                nat=set(
                    row[[f'nationality_{i}_iso_3166_1_alpha_2' for i in range(1, 4)]]
                    .dropna()
                    .astype(str)
                ),
                por=set(
                    row[[f'place_of_residence_{i}_bfs_gmde_code_latest' for i in range(1, 3)]]
                    .dropna()
                    .astype(str)
                ),
                gen=row.gender,
            )
            for r_idx, row in subset_df.iterrows()]

    def _enrich_person_data(self, person: Person, founder: Person = None):
        """
        Enriches a person's data by computing tokens and overlaps relative to a founder.
        """
        person.fn_tokens = set(person.fn.split())
        person.ln_tokens = set(person.ln.split())
        person.name_token_set = person.fn_tokens | person.ln_tokens

        if founder:
            person.hometown_overlap = person.ht & founder.ht
            person.nationality_overlap = person.nat & founder.nat
            person.residence_overlap = person.por & founder.por
            person.location_match_flag = bool(person.hometown_overlap if 'CH' in person.nationality_overlap
                                              else person.nationality_overlap | person.residence_overlap)

        return person

    def _create_return_dataframe(self, clusters: list[Person]):
        """
        Creates a DataFrame from a list of clustered Person objects, joining back with the original data.
        """
        mapped_df = pd.DataFrame([asdict(person) for person in clusters])
        mapped_df.set_index('idx', inplace=True)
        return self.df.drop(columns=['fid']).join(mapped_df[['heuristic', 'fid']])

    def _prepare_ordered_names(self):
        """
        Prepares sorted concatenation of names to aid clustering across companies.
        """
        self.within_clusters['ordered_names'] = (
            self.within_clusters.apply(
                lambda row: ' '.join(sorted(
                    row['first_name_norm'].split() + row['last_name_norm'].split()
                )),
                axis=1
            )
        )

    def _prepare_location_codes(self):
        """
        Prepares combined geographical and nationality codes for clustering purposes.
        """
        cols_dict = {
            'hometown_codes': [f'hometown_{i}_bfs_gmde_code_latest' for i in range(1, 6)],
            'nationality_codes': [f'nationality_{i}_iso_3166_1_alpha_2' for i in range(1, 4)],
            'place_of_residence_codes': [f'place_of_residence_{i}_bfs_gmde_code_latest' for i in range(1, 3)]
        }

        for key, cols in cols_dict.items():
            self.within_clusters[key] = self.within_clusters[cols].apply(
                lambda row: ', '.join(sorted(map(str, row.dropna()))), axis=1
            )

    def _split_swiss_foreigners(self):
        """
        Splits data into Swiss nationals and foreigners for specialized clustering.
        """
        swiss = self.within_clusters[self.within_clusters.nationality_codes.str.contains('CH')].copy()
        foreigners = self.within_clusters[~self.within_clusters.nationality_codes.str.contains('CH')].copy()
        return swiss, foreigners

    def _merge_swiss_entries(self, swiss: pd.DataFrame) -> dict:
        """
        Merges Swiss person entries based on names and hometown codes.
        """
        swiss_no_duplicates = swiss.drop_duplicates(subset=['ehraid', 'ordered_names', 'hometown_codes'])
        swiss_merged = swiss_no_duplicates.merge(
            swiss_no_duplicates[['ordered_names', 'hometown_codes', 'fid']].rename(columns={'fid': 'fid_across'}),
            on=['ordered_names', 'hometown_codes']
        )

        swiss_merged = swiss_merged[swiss_merged.fid != swiss_merged.fid_across]
        mask = swiss_merged.fid != swiss_merged.fid_across
        swiss_merged.loc[mask, 'fid_across'] = swiss_merged.loc[mask, ['fid', 'fid_across']].min(axis=1)

        swiss_merged = swiss_merged[swiss_merged.fid != swiss_merged.fid_across].drop_duplicates(
            subset=['ehraid', 'ordered_names', 'hometown_codes', 'fid']
        )

        return {k: v for k, v in zip(swiss_merged.fid, swiss_merged.fid_across) if pd.notna(k) and pd.notna(v)}

    def _merge_foreign_entries(self, foreigners: pd.DataFrame) -> dict:
        """
        Merges foreign person entries based on names, nationality, and residence codes.
        """
        foreigners_no_duplicates = foreigners.drop_duplicates(
            subset=['ehraid', 'ordered_names', 'nationality_codes', 'place_of_residence_codes']
        )
        foreigners_merged = foreigners_no_duplicates.merge(
            foreigners_no_duplicates[['ordered_names', 'nationality_codes', 'place_of_residence_codes', 'fid']]
            .rename(columns={'fid': 'fid_across'}),
            on=['ordered_names', 'nationality_codes', 'place_of_residence_codes']
        )

        foreigners_merged = foreigners_merged[foreigners_merged.fid != foreigners_merged.fid_across]
        mask = foreigners_merged.fid != foreigners_merged.fid_across
        foreigners_merged.loc[mask, 'fid_across'] = foreigners_merged.loc[mask, ['fid', 'fid_across']].min(axis=1)

        foreigners_merged = foreigners_merged[foreigners_merged.fid != foreigners_merged.fid_across].drop_duplicates(
            subset=['ehraid', 'ordered_names', 'nationality_codes', 'place_of_residence_codes', 'fid']
        )

        return {k: v for k, v in zip(foreigners_merged.fid, foreigners_merged.fid_across) if pd.notna(k) and pd.notna(v)}

    def _update_fids(self, fid2fid_across: dict):
        """
        Updates the fid identifiers according to merged clusters.
        """
        self.within_clusters['fid'] = self.within_clusters.fid.replace(fid2fid_across)

    def _normalize_fids(self):
        """
        Normalizes fid identifiers into a sequential numerical range.
        """
        unique_fids = sorted(self.within_clusters['fid'].dropna().unique())
        fid_mapping = {fid: i + 1 for i, fid in enumerate(unique_fids)}
        self.within_clusters['fid'] = self.within_clusters.fid.replace(fid_mapping)

    def cluster_within(self) -> pd.DataFrame:
        """
        Clusters persons within companies based on heuristics detailed in Table 1.

        Returns:
            pd.DataFrame: DataFrame with clustered identifiers (fid) for each person within companies.
        """
        all_clusters = []
        company_groups = self.df.groupby('ehraid')

        for _, company in tqdm(company_groups, desc='Cluster people within company'):
            # Initialize and enrich founders data
            founders = self._initialize_clusters(company, founders=True)

            # Pre-process all founders
            for founder in founders:
                self._enrich_person_data(founder)

            # First pass: merge founders that match with each other
            for i in range(len(founders) - 1):
                for j in range(i + 1, len(founders)):
                    for h_idx, heuristic in enumerate(self.heuristics_within):
                        if heuristic(founders[i], founders[j]):
                            # Update founder j's fid with founder i's
                            founders[j].fid = founders[i].fid
            all_clusters.extend(founders)

            # Process each shab entry to find matches
            shab_groups = company.groupby('shab_id')
            for _, shab_entry in shab_groups:
                people = self._initialize_clusters(shab_entry, founders=False)
                if not people:
                    continue

                # Try each founder against each person, and only keep the highest confidence match
                for founder in founders:
                    best_match = None
                    best_heuristic = None

                    for person in people:
                        # Enrich the person with computed values relative to this founder
                        self._enrich_person_data(person, founder)
                        # Try each heuristic in order of confidence
                        for h_idx, heuristic in enumerate(self.heuristics_within):
                            if heuristic(founder, person):
                                # If we found a match with a better heuristic than before
                                if best_match is None or h_idx < best_heuristic:
                                    best_match = person
                                    best_heuristic = h_idx + 1
                                break  # No need to try lower confidence heuristics for this person

                    # If we found a match for this founder, mark it and add to results
                    if best_match:
                        if not best_match.fid:  # Person hasn't been mapped
                            best_match.fid = founder.fid
                        elif best_match.fid and best_heuristic < best_match.heuristic:  # Person fits better to the current founder if multiple matches
                            best_match.fid = founder.fid
                            best_match.heuristic = best_heuristic

                all_clusters.extend(people)

        return self._create_return_dataframe(all_clusters)

    def cluster_across(self) -> pd.DataFrame:
        """
        Clusters persons across companies by merging previously clustered fids using high-confidence heuristics.

        Returns:
            pd.DataFrame: Final clustered DataFrame with normalized fid across companies.
        """
        self._prepare_ordered_names()
        self._prepare_location_codes()

        swiss, foreigners = self._split_swiss_foreigners()

        fid2fid_across = {}
        fid2fid_across.update(self._merge_swiss_entries(swiss))
        fid2fid_across.update(self._merge_foreign_entries(foreigners))

        self._update_fids(fid2fid_across)
        self._normalize_fids()

        return self.within_clusters.drop(columns=[
            'ordered_names', 'hometown_codes',
            'nationality_codes', 'place_of_residence_codes'
        ])

    def cluster(self):
        """
        Performs clustering of individuals both within and across companies, producing the final cluster identifiers.

        Returns:
            pd.DataFrame: DataFrame with final clustering results.
        """
        self.within_clusters = self.cluster_within()
        return self.cluster_across()
