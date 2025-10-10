import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

import useful_rdkit_utils as uru
from chembl_structure_pipeline import standardizer, checker

uru.rd_shut_the_hell_up()

import sys
from typing import List, Optional

import pystow
from tqdm.auto import tqdm
from rdkit.Chem.rdchem import Mol


class REOS:
    """REOS - Rapid Elimination Of Swill. Adapted to fit our needs with more rule info.\n
    Walters, Ajay, Murcko, "Recognizing molecules with druglike properties"\n
    Curr. Opin. Chem. Bio., 3 (1999), 384-387\n
    https://doi.org/10.1016/S1367-5931(99)80058-1
    """

    def __init__(self, active_rules: Optional[List[str]] = None) -> None:
        """
        Initialize the REOS class.

        :param active_rules: List of active rules. If None, the default rule 'Glaxo' is used.
        :type active_rules: Optional[List[str]]
        :default active_rules: None
        """
        self.output_smarts = False
        if active_rules is None:
            active_rules = ['Glaxo']
        url = 'https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv'
        self.rule_path = pystow.ensure('useful_rdkit_utils', 'data', url=url)
        self.active_rule_df = None
        self.rule_df = pd.read_csv(self.rule_path)
        self.read_rules(self.rule_path, active_rules)

    def set_output_smarts(self, output_smarts):
        """Determine whether SMARTS are returned
        :param output_smarts: True or False
        :return: None
        """
        self.output_smarts = output_smarts

    def parse_smarts(self):
        """Parse the SMARTS strings in the rules file to molecule objects and check for validity

        :return: True if all SMARTS are parsed, False otherwise
        """
        smarts_mol_list = []
        smarts_are_ok = True
        for idx, smarts in enumerate(self.rule_df.smarts, 1):
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                smarts_are_ok = False
                print(f"Error processing SMARTS on line {idx}", file=sys.stderr)
            smarts_mol_list.append(mol)
        self.rule_df['pat'] = smarts_mol_list
        return smarts_are_ok

    def read_rules(self, rules_file, active_rules=None):
        """Read a rules file

        :param rules_file: name of the rules file
        :param active_rules: list of active rule sets, all rule sets are used if
            this is None
        :return: None
        """
        if self.parse_smarts():
            self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules")
            if len(self.active_rule_df) == 0:
                available_rules = sorted(list(self.rule_df["rule_set_name"].unique()))
                raise ValueError(f"Supplied rules: {active_rules} not available. Please select from {available_rules}")

        else:
            print("Error reading rules, please fix the SMARTS errors reported above", file=sys.stderr)
            sys.exit(1)
        if active_rules is not None:
            self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules").copy()
        else:
            self.active_rule_df = self.rule_df.copy()

    def set_active_rule_sets(self, active_rules=None):
        """Set the active rule set(s)

        :param active_rules: list of active rule sets
        :return: None
        """
        assert active_rules
        self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules")

    def set_min_priority(self, min_priority: int) -> None:
        """Set the minimum priority for rules to be included in the active rule set.

        :param min_priority: The minimum priority for rules to be included.
        :return: None
        """
        # reset active_rule_df
        self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules").copy()
        # filter to only include rules with priority greater than or equal to min_priority
        self.active_rule_df = self.active_rule_df.query("priority >= @min_priority")

    def get_available_rule_sets(self):
        """Get the available rule sets in rule_df

        :return: a list of available rule sets
        """
        return self.rule_df.rule_set_name.unique()

    def get_active_rule_sets(self):
        """Get the active rule sets in active_rule_df

        :return: a list of active rule sets
        """
        return self.active_rule_df.rule_set_name.unique()

    def drop_rule(self, description: str) -> None:
        """Drops a rule from the active rule set based on its description.

        :param: description: The description of the rule to be dropped.
        :return: None
        """
        num_rules_before = len(self.active_rule_df)
        self.active_rule_df = self.active_rule_df.query("description != @description")
        num_rules_after = len(self.active_rule_df)
        print(f"Dropped {num_rules_before - num_rules_after} rule(s)")

    def get_rule_file_location(self):
        """Get the path to the rules file as a Path

        :return: Path for rules file
        """
        return self.rule_path

    def process_mol(self, mol, detailed: bool = False):
        """
        Match a molecule against the active rule set.

        :param mol: input RDKit molecule
        :param detailed: if True, returns additional info regarding all failed rules.
                        If False (default), returns only the first failed rule or "ok".
        :return:
            - If detailed is False:
                returns a tuple (rule_set_name, description) (or with smarts if output_smarts is True),
                or ("ok", "ok") (or ("ok", "ok", "ok")) if no rule is failed.
            - If detailed is True:
                returns a flattened tuple:
                    * If self.output_smarts is False: (rule_set_name, description, num_failed, failed_rules)
                    * If self.output_smarts is True: (rule_set_name, description, smarts, num_failed, failed_rules)
        """
        cols = ['description', 'rule_set_name', 'smarts', 'pat', 'max']
        violations = []
        for desc, rule_set_name, smarts, pat, max_val in self.active_rule_df[cols].values:
            if len(mol.GetSubstructMatches(pat)) > max_val:
                if self.output_smarts:
                    violation = (rule_set_name, desc, smarts)
                else:
                    violation = (rule_set_name, desc)
                violations.append(violation)

        if detailed:
            num_failed = len(violations)
            if violations:
                first_violation = violations[0]
            else:
                # Use the default "ok" values based on whether SMARTS are output.
                first_violation = ("ok", "ok", "ok") if self.output_smarts else ("ok", "ok")
            # Flatten the tuple so that the returned value has the expected number of elements.
            if self.output_smarts:
                # Expecting 5 columns: rule_set_name, description, smarts, num_failed, failed_rules
                return first_violation[0], first_violation[1], first_violation[2], num_failed, violations
            else:
                # Expecting 4 columns: rule_set_name, description, num_failed, failed_rules
                return first_violation[0], first_violation[1], num_failed, violations
        else:
            if violations:
                return violations[0]
            else:
                return ("ok", "ok", "ok") if self.output_smarts else ("ok", "ok")


    def process_smiles(self, smiles, detailed: bool = False):
        """
        Convert SMILES to an RDKit molecule and call process_mol.
        
        :param smiles: input SMILES string
        :param detailed: if True, returns additional detailed info from process_mol.
        :return: the result from process_mol, or None if the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error parsing SMILES {smiles}")
            return None
        return self.process_mol(mol, detailed=detailed)

    def pandas_smiles(self, smiles_list: List[str], detailed: bool = False) -> pd.DataFrame:
        """
        Process a list of SMILES strings and return a DataFrame with the results.

        :param smiles_list: list of SMILES strings
        :param detailed: if True, the DataFrame includes two extra columns:
                        'num_failed' (the number of violated rules) and 
                        'failed_rules' (the list of rules that failed).
        :return: pandas DataFrame with the results.
        """
        results = []
        for smiles in tqdm(smiles_list):
            results.append(self.process_smiles(smiles, detailed=detailed))
        
        if detailed:
            if self.output_smarts:
                column_names = ['rule_set_name', 'description', 'smarts', 'num_failed', 'failed_rules']
            else:
                column_names = ['rule_set_name', 'description', 'num_failed', 'failed_rules']
        else:
            if self.output_smarts:
                column_names = ['rule_set_name', 'description', 'smarts']
            else:
                column_names = ['rule_set_name', 'description']
        
        return pd.DataFrame(results, columns=column_names)



class SMILESCleaner:
    def __init__(self, smiles: List[str]):
        self.dataframe = pd.DataFrame()
        self.dataframe['smiles'] = smiles
        self.dataframe['valid'] = True
        self.dataframe['comment'] = None
        self.dataframe['n_reos_warnings'] = 0
        self.dataframe['inchi'] = None

        self.ring_system_lookup = uru.RingSystemLookup.default()
        self.reos = REOS(["Dundee"])
        
        self.canonicalize_all()
        self.deduplicate_all()
        self.standardize_all()
        self.reos_filter_all()
        
    def canonicalize(self, smiles: str) -> str:
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        return self.mol_to_smiles(mol), self.mol_to_inchi(mol)
    
    def canonicalize_all(self) -> None:
        canon = self.dataframe['smiles'].apply(self.canonicalize)
        canon_smiles = canon.apply(lambda x: x[0] if x is not None else None)
        canon_inchi = canon.apply(lambda x: x[1] if x is not None else None)
        # Where the canonicalization failed, the original SMILES is invalid
        canon_worked = canon_smiles.notnull() & canon_inchi.notnull()
        self.dataframe.loc[canon_worked, 'smiles'] = canon_smiles[canon_worked]
        self.dataframe.loc[canon_worked, 'inchi'] = canon_inchi[canon_worked]
        self.dataframe.loc[~canon_worked, 'valid'] = False
        self.dataframe.loc[~canon_worked, 'comment'] = 'Canonicalization failed'
        
        # print("Canonicalization done")
        
    def deduplicate_all(self ) -> None:
        duplicate_smiles = self.dataframe.duplicated(subset='smiles', keep='first')
        duplicate_inchis = self.dataframe.duplicated(subset='inchi', keep='first')
        duplicates = duplicate_smiles | duplicate_inchis
        self.dataframe.loc[duplicates, 'valid'] = False
        self.dataframe.loc[duplicates, 'comment'] = 'Duplicate'
        
        # print("Deduplication done")
        
    @staticmethod   
    def smiles_to_mol(smiles: str) -> Chem.rdchem.Mol:
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
        
    @staticmethod   
    def mol_to_molblock(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        try:
            return Chem.MolToMolBlock(mol)
        except:
            return None
        
    @staticmethod  
    def molblock_to_mol(molblock: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        try:
            return Chem.MolFromMolBlock(molblock)
        except:
            return None
        
    @staticmethod  
    def mol_to_smiles(mol: Chem.rdchem.Mol) -> str:
        try:
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    @staticmethod  
    def mol_to_inchi(mol: Chem.rdchem.Mol) -> str:
        try:
            return Chem.MolToInchi(mol)
        except:
            return None
        
    @staticmethod   
    def standardize(molblock: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        try:
            molblock = standardizer.standardize_molblock(molblock)
            # molblock = standardizer.get_parent_molblock(molblock)
            return molblock
        except:
            return None
        
    @staticmethod
    def standardize_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        """ Adapted from https://www.blopig.com/blog/2022/05/molecular-standardization/
        
        Standardize the RDKit molecule, select its parent molecule, uncharge it, 
        then enumerate all the tautomers.
        If verbose is true, an explanation of the steps and structures of the molecule
        as it is standardized will be output."""
        # Follows the steps from:
        #  https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg Landrum) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ -- thanks JP!
        try:
            # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
            clean_mol = rdMolStandardize.Cleanup(mol) 

            # if many fragments, get the "parent" (the actual mol we are interested in) 
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

            # try to neutralize molecule
            uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
            uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

            # Note: no attempt is made at reionization at this step
            # nor ionization at some pH (RDKit has no pKa caculator);
            # the main aim to to represent all molecules from different sources
            # in a (single) standard way, for use in ML, catalogues, etc.
            te = rdMolStandardize.TautomerEnumerator() # idem
            taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
            return taut_uncharged_parent_clean_mol
        
        except:
            return None

    @staticmethod
    def structure_check(molblock: Chem.rdchem.Mol) -> int:
        """
        The checker assesses the quality of a structure. It highlights specific features or comment in the structure that may need to be revised. Together with the description of the issue, the checker process returns a penalty score (between 0-9) which reflects the seriousness of the issue (the higher the score, the more critical is the issue)
        """
        try:
            return checker.check_molblock(molblock)
        except:
            return None
        
    def standardize_all(self) -> None:
        molblocks = self.dataframe['smiles'].apply(self.smiles_to_mol).apply(self.standardize_mol).apply(self.mol_to_molblock)
        std_molblocks = molblocks.apply(self.standardize)
        std_mol = std_molblocks.apply(self.molblock_to_mol)
        std_smiles = std_mol.apply(self.mol_to_smiles)
        comment = std_molblocks.apply(self.structure_check)
        
        # Where the standardization failed, the original SMILES is invalid
        std_worked = std_smiles.notnull()
        self.dataframe.loc[std_worked, 'smiles'] = std_smiles[std_worked]
        self.dataframe.loc[~std_worked, 'valid'] = False
        self.dataframe.loc[~std_worked, 'comment'] = 'Standardization failed'
        self.dataframe.loc[std_worked, 'comment'] = comment[std_worked]
        
        # print("Standardization done")
        
    def reos_filter_all(self) -> None:
        reos_df = self.reos.pandas_smiles(self.dataframe['smiles'], detailed=True)
        ok = pd.Series([True if x == "ok" else False for x in reos_df['description']])
        # reos_msg = pd.Series([f"REOS filter failed: {x}" for x in reos_df['description']])
        
        reos_msg = []
        for rules in reos_df['failed_rules']:
            if rules is None:
                reos_msg.append(None)
                continue
            error_list = []
            for rule in rules:
                r, e = rule
                error_list.append(e)
            error_str = ', '.join(error_list)
            reos_msg.append(error_str)      
        reos_msg = pd.Series(reos_msg)
          
        # self.dataframe.loc[~ok, 'valid'] = False
        self.dataframe.loc[~ok, 'comment'] = reos_msg[~ok]
        self.dataframe.loc[~ok, 'n_reos_warnings'] = reos_df['num_failed'][~ok]
        
        # print("REOS filter done")
        
    def get_valid(self) -> pd.DataFrame:
        return self.dataframe[self.dataframe['valid']].sort_values('n_reos_warnings', ascending=True)
    
    def get_data(self) -> pd.DataFrame:
        return self.dataframe