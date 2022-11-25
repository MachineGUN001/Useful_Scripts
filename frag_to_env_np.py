#!/usr/bin/env python3
#==============================================================================
# author          : Pavel Polishchuk
# date            : 31-08-2018
# version         :
# python_version  :
# copyright       : Pavel Polishchuk 2018
# license         :
#==============================================================================
__author__ = 'pavel'

import argparse
import sys

from itertools import permutations
from multiprocessing import Pool, cpu_count
from rdkit import Chem

#from .mol_context import get_std_context_core_permutations

import re
from itertools import product, permutations, combinations
from collections import defaultdict
from rdkit import Chem
#from .functions import mol_to_smarts


import re
import sys
from rdkit import Chem


def mol_to_smarts(mol, keep_h=True):
    # keep_h - will increase the count of H atoms for atoms with attached hydrogens to create a valid smarts
    # e.g. [H]-[CH2]-[*] -> [H]-[CH3]-[*]

    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache()

    # change the isotope to 42
    for atom in mol.GetAtoms():
        if keep_h:
            s = sum(na.GetAtomicNum() == 1 for na in atom.GetNeighbors())
            if s:
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + s)
        atom.SetIsotope(42)

    # print out the smiles - all the atom attributes will be fully specified
    smarts = Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True)
    # remove the 42 isotope labels
    smarts = re.sub(r'\[42', "[", smarts)

    return smarts


def smiles_to_smarts(smi, keep_h=True):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        sys.stderr.write("Can't generate mol for: %s\n" % smi)
        return None
    return mol_to_smarts(mol, keep_h)



__author__ = 'pavel'

patt_remove_map = re.compile("\[\*\:[0-9]+\]")   # to change CC([*:1])O to CC([*])O
patt_remove_h = re.compile("(?<!\[)H[1-9]*(?=:[0-9])")   # to remove H after atoms with maps: [CH2:1] to [C:1], but not touching [H] or [nH]


def __get_submol(mol, atom_ids):
    bond_ids = []
    for pair in combinations(atom_ids, 2):
        b = mol.GetBondBetweenAtoms(*pair)
        if b:
            bond_ids.append(b.GetIdx())
    m = Chem.PathToSubmol(mol, bond_ids)
    m.UpdatePropertyCache()
    return m


def __bonds_to_atoms(mol, bond_ids):
    output = []
    for i in bond_ids:
        b = mol.GetBondWithIdx(i)
        output.append(b.GetBeginAtom().GetIdx())
        output.append(b.GetEndAtom().GetIdx())
    return tuple(set(output))


def __get_context_env(mol, radius):
    """
    INPUT:
        mol - Mol object containing chain(s) of molecular context
        radius - integer, number of bonds to cut context
    OUTPUT:
        Mol containing only atoms within the specified radius from the attachment point(s).
        All explicit Hs will be stripped.
    """
    # mol is context consisting of one or more groups with single attachment point

    m = Chem.RemoveHs(mol)
    m = Chem.RWMol(m)

    bond_ids = set()
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            i = radius
            b = Chem.FindAtomEnvironmentOfRadiusN(m, i, a.GetIdx())
            while not b and i > 0:
                i -= 1
                b = Chem.FindAtomEnvironmentOfRadiusN(m, i, a.GetIdx())
            bond_ids.update(b)

    atom_ids = set(__bonds_to_atoms(m, bond_ids))

    dummy_atoms = []

    for a in m.GetAtoms():
        if a.GetIdx() not in atom_ids:
            nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
            intersect = nei_ids & atom_ids
            if intersect:
                dummy_atom_bonds = []
                for ai in intersect:
                    dummy_atom_bonds.append((ai, m.GetBondBetweenAtoms(a.GetIdx(), ai).GetBondType()))
                dummy_atoms.append(dummy_atom_bonds)

    for data in dummy_atoms:
        dummy_id = m.AddAtom(Chem.Atom(0))
        for atom_id, bond_type in data:
            m.AddBond(dummy_id, atom_id, bond_type)
        atom_ids.add(dummy_id)

    m = __get_submol(m, atom_ids)

    return m


def __replace_att(mol, repl_dict):
    for a in mol.GetAtoms():
        map_num = a.GetAtomMapNum()
        if map_num in repl_dict:
            a.SetAtomMapNum(repl_dict[map_num])


def __get_maps_and_ranks(env, keep_stereo=False):
    """
    Return the list of attachment point map numbers and
    the list of canonical SMILES without mapped attachment points (ranks)
    """
    tmp_mol = Chem.Mol(env)
    maps = []
    ranks = []
    for comp in Chem.GetMolFrags(tmp_mol, asMols=True, sanitizeFrags=False):
        for a in comp.GetAtoms():
            atom_num = a.GetAtomMapNum()
            if atom_num:
                maps.append(atom_num)
                a.SetAtomMapNum(0)
                break
        ranks.append(Chem.MolToSmiles(comp, isomericSmiles=keep_stereo))
    return maps, ranks


def __standardize_att_by_env(env, core, keep_stereo=False):
    """
    Set attachment point numbers in core and context according to canonical ranks of attachment points in context
    Ties are broken
    Makes changes in place
    """
    maps, ranks = __get_maps_and_ranks(env, keep_stereo)
    new_att = {m: i+1 for i, (r, m) in enumerate(sorted(zip(ranks, maps)))}
    __replace_att(core, new_att)
    __replace_att(env, new_att)


def __get_att_permutations(env):
    """
    Return possible permutations of attachment point map numbers as a tuple of dicts,
    where each dict: key - old number, value - new number
    """
    maps, ranks = __get_maps_and_ranks(env)

    d = defaultdict(list)
    for rank, att in zip(ranks, maps):
        d[rank].append(att)

    c = []
    for v in d.values():
        c.append([dict(zip(v, x)) for x in permutations(v, len(v))])

    return tuple(__merge_dicts(*item) for item in product(*c))


def __permute_att(mol, d):
    new_mol = Chem.Mol(mol)
    for a in new_mol.GetAtoms():
        i = a.GetAtomMapNum()
        if i in d:
            a.SetAtomMapNum(d[i])
    return new_mol


def __merge_dicts(*dicts):
    res = dicts[0].copy()
    for item in dicts[1:]:
        res.update(item)
    return res


def __standardize_smiles_with_att_points(mol, keep_stereo=False):
    """
    to avoid different order of atoms in SMILES with different map number of attachment points

    smi = ["ClC1=C([*:1])C(=S)C([*:2])=C([*:3])N1",
           "ClC1=C([*:1])C(=S)C([*:3])=C([*:2])N1",
           "ClC1=C([*:2])C(=S)C([*:1])=C([*:3])N1",
           "ClC1=C([*:2])C(=S)C([*:3])=C([*:1])N1",
           "ClC1=C([*:3])C(=S)C([*:1])=C([*:2])N1",
           "ClC1=C([*:3])C(=S)C([*:2])=C([*:1])N1"]

    these will produce different output with RDKit MolToSmiles():
        S=c1c([*:1])c(Cl)[nH]c([*:3])c1[*:2]
        S=c1c([*:1])c(Cl)[nH]c([*:2])c1[*:3]
        S=c1c([*:1])c([*:3])[nH]c(Cl)c1[*:2]
        S=c1c([*:2])c(Cl)[nH]c([*:1])c1[*:3]
        S=c1c([*:1])c([*:2])[nH]c(Cl)c1[*:3]
        S=c1c([*:2])c([*:1])[nH]c(Cl)c1[*:3]

    output of this function
        S=c1c([*:2])c([*:3])[nH]c(Br)c1[*:1]
        S=c1c([*:3])c([*:2])[nH]c(Br)c1[*:1]
        S=c1c([*:1])c([*:3])[nH]c(Br)c1[*:2]
        S=c1c([*:3])c([*:1])[nH]c(Br)c1[*:2]
        S=c1c([*:1])c([*:2])[nH]c(Br)c1[*:3]
        S=c1c([*:2])c([*:1])[nH]c(Br)c1[*:3]

    https://sourceforge.net/p/rdkit/mailman/message/35862258/
    """

    # update property cache if needed
    if mol.NeedsUpdatePropertyCache():
        mol.UpdatePropertyCache()

    # store original maps and remove map numbers from mol
    backup_atom_map = "backupAtomMap"
    for a in mol.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    # get canonical ranks for atoms for a mol without maps
    atoms = list(zip(list(Chem.CanonicalRankAtoms(mol)), [a.GetIdx() for a in mol.GetAtoms()]))
    atoms.sort()

    # set new atom maps based on canonical order
    rep = {}
    atom_map = 1
    for pos, atom_idx in atoms:
        a = mol.GetAtomWithIdx(atom_idx)
        if a.HasProp(backup_atom_map):
            a.SetAtomMapNum(atom_map)
            rep["[*:%i]" % atom_map] = "[*:%i]" % a.GetIntProp(backup_atom_map)
            atom_map += 1

    # get SMILES and relabel with original map numbers
    s = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo)
    rep = dict((re.escape(k), v) for k, v in rep.items())
    patt = re.compile("|".join(rep.keys()))
    s = patt.sub(lambda m: rep[re.escape(m.group(0))], s)

    return s


def get_std_context_core_permutations(context, core, radius, keep_stereo):
    """
    INPUT:
        context - Mol or SMILES containing full chain(s) of a context with labeled attachment point(s),
                  if context is absent (e.g.for radius 0) specify empty string or empty Mol
        core    - Mol or SMILES of a core fragment with labeled attachment point(s)
        keep_stereo - boolean to keep stereo information in output
        radius  - integer (0, 1, 2, etc), number of bonds to cut context
    OUTPUT:
        SMILES of a context environment of a specified radius,
        list of SMILES of a core fragment with possible permutations of attachment point numbers
        env_smi, (core_smi_1, core_smi_2, ...)

        env_smi will not contain any Hs

        for radius 0 attachment point numbers will be stripped, but the string will correspond to core SMILES with
        radius > 0 if remove all map numbers from SMILES

    Output SMILES are standardized
    """

    if isinstance(context, str):
        context = Chem.MolFromSmiles(context)
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)

    # remove Hs from context and core
    if context:  # context cannot be H (no check needed), if so the user will obtain meaningless output
        context = Chem.RemoveHs(context)
    if core and Chem.MolToSmiles(core) != '[H][*:1]':
        core = Chem.RemoveHs(core)

    if radius == 0 and core:

        if not keep_stereo:
            Chem.RemoveStereochemistry(core)

        s = __standardize_smiles_with_att_points(core, keep_stereo)
        s = patt_remove_map.sub("[*]", s)

        return '', (s, )

    if core and context:

        att_num = len(Chem.GetMolFrags(context))

        if not keep_stereo:
            Chem.RemoveStereochemistry(context)
            Chem.RemoveStereochemistry(core)

        env = __get_context_env(context, radius)   # cut context to radius
        __standardize_att_by_env(env, core, keep_stereo)
        env_smi = Chem.MolToSmiles(env, isomericSmiles=keep_stereo, allBondsExplicit=True)

        if att_num == 1:

            return env_smi, (__standardize_smiles_with_att_points(core, keep_stereo), )

        else:

            res = []
            p = __get_att_permutations(env)

            # permute attachment point numbering only in core,
            # since permutations in env will give the same canonical smiles
            if len(p) > 1:
                for d in p:
                    c = __permute_att(core, d)
                    res.append(c)
            else:
                res.append(core)

            # get distinct standardized SMILES
            d = tuple(set(__standardize_smiles_with_att_points(m, keep_stereo) for m in res))

            return env_smi, d

    return None, None


def get_canon_context_core(context, core, radius, keep_stereo=False):
    # context and core are Mols or SMILES
    # returns SMILES by default
    res = get_std_context_core_permutations(context, core, radius, keep_stereo)
    if res:
        env, cores = res
        return env, sorted(cores)[0]
    else:
        return None, None


def combine_core_env_to_rxn_smarts(core, env, keep_h=True):

    if isinstance(env, str):
        m_env = Chem.MolFromSmiles(env, sanitize=False)
    if isinstance(core, str):
        m_frag = Chem.MolFromSmiles(core, sanitize=False)

    backup_atom_map = "backupAtomMap"

    # put all atom maps to atom property and remove them
    for a in m_env.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    for a in m_frag.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    # set canonical ranks for atoms in env without maps
    m_env.UpdatePropertyCache()
    for atom_id, rank in zip([a.GetIdx() for a in m_env.GetAtoms()], list(Chem.CanonicalRankAtoms(m_env))):
        a = m_env.GetAtomWithIdx(atom_id)
        if not a.HasProp(backup_atom_map):
            a.SetAtomMapNum(rank + 1)  # because ranks start from 0

    m = Chem.RWMol(Chem.CombineMols(m_frag, m_env))

    links = defaultdict(list)  # pairs of atom ids to create bonds
    att_to_remove = []  # ids of att points to remove
    for a in m.GetAtoms():
        if a.HasProp(backup_atom_map):
            i = a.GetIntProp(backup_atom_map)
            links[i].append(a.GetNeighbors()[0].GetIdx())
            att_to_remove.append(a.GetIdx())

    for i, j in links.values():
        m.AddBond(i, j, Chem.BondType.SINGLE)

    for i in sorted(att_to_remove, reverse=True):
        m.RemoveAtom(i)

    comb_sma = mol_to_smarts(m, keep_h)
    if not keep_h:  # remove H only in mapped env part
        comb_sma = patt_remove_h.sub('', comb_sma)
    return comb_sma

def process_line(line):
    # returns env_smi, core_smi, heavy_atoms_num, core_smarts

    output = []
    smi, id, core, context = line.strip().split(',')

    if (not core and not context) or (_keep_mols and id not in _keep_mols):
        return output
    else:
        # one split
        if not core:
            residues = context.split('.')
            if len(residues) == 2:
                for context, core in permutations(residues, 2):
                    if context == '[H][*:1]':   # ignore such cases
                        continue
                    mm = Chem.MolFromSmiles(core, sanitize=False)
                    num_heavy_atoms = mm.GetNumHeavyAtoms() if mm else float('inf')
                    if num_heavy_atoms <= _max_heavy_atoms:
                        env, cores = get_std_context_core_permutations(context, core, _radius, _keep_stereo)
                        if env and cores:
                            # for 1 cut cores will always contain 1 item
                            if not _store_comp_id:
                                output.append((env, cores[0], num_heavy_atoms))
                            else:
                                output.append((env, cores[0], num_heavy_atoms, id))
            else:
                sys.stderr.write('more than two fragments in context (%s) where core is empty' % context)
                sys.stderr.flush()
        # two or more splits
        else:
            mm = Chem.MolFromSmiles(core, sanitize=False)
            num_heavy_atoms = mm.GetNumHeavyAtoms() if mm else float('inf')
            if num_heavy_atoms <= _max_heavy_atoms:
                env, cores = get_std_context_core_permutations(context, core, _radius, _keep_stereo)
                if env and cores:
                    for c in cores:
                        if not _store_comp_id:
                            output.append((env, c, num_heavy_atoms))
                        else:
                            output.append((env, c, num_heavy_atoms, id))
        return output


def init(keep_mols, radius, keep_stereo, max_heavy_atoms, store_comp_id):
    global _keep_mols
    global _radius
    global _keep_stereo
    global _max_heavy_atoms
    global _store_comp_id
    _keep_mols = set([line.strip() for line in open(keep_mols).readlines()]) if keep_mols else set()
    _radius = radius
    _keep_stereo = keep_stereo
    _max_heavy_atoms = max_heavy_atoms
    _store_comp_id = store_comp_id


def main(input_fname, output_fname, keep_mols, radius, keep_stereo, max_heavy_atoms, ncpu, store_comp_id, verbose):

    # radius and remove_stereo are supplied to process_context_core via global environment (ugly but working solution)

    ncpu = min(cpu_count(), max(ncpu, 1))
    p = Pool(ncpu, initializer=init, initargs=(keep_mols, radius, keep_stereo, max_heavy_atoms, store_comp_id))

    try:
        with open(output_fname, 'wt') as out:

            with open(input_fname) as f:

                for i, res in enumerate(p.imap_unordered(process_line, f, chunksize=1000), 1):

                    for item in res:
                        if item:
                            out.write(','.join(map(str, item)) + '\n')

                    if verbose and i % 1000 == 0:
                        sys.stderr.write('\r%i lines passed' % i)
                        sys.stderr.flush()

    finally:
        p.close()


def entry_point():
    parser = argparse.ArgumentParser(description='Create text file for fragment replacement from fragmented molecules '
                                                 'obtained with fragmentation.py. '
                                                 'The output may contain duplicated lines which should be filtered out '
                                                 'externally.')
    parser.add_argument('-i', '--input', metavar='frags.txt', required=True,
                        help='fragmented molecules.')
    parser.add_argument('-o', '--out', metavar='output.txt', required=True,
                        help='output text file.')
    parser.add_argument('-k', '--keep_mols', metavar='molnames.txt', required=False, default=None,
                        help='file with mol names to keep. Molecules which are not in the list will be ignored.')
    parser.add_argument('-r', '--radius', metavar='NUMBER', required=False, default=1,
                        help='radius of molecular context (in bonds) which will be taken into account. Default: 1.')
    parser.add_argument('-a', '--max_heavy_atoms', metavar='NUMBER', required=False, default=20,
                        help='maximum number of heavy atoms in cores. If the number of atoms exceeds the limit '
                             'fragment will be discarded. Default: 20.')
    parser.add_argument('-s', '--keep_stereo', action='store_true', default=False,
                        help='set this flag if you want to keep stereo in context and core parts.')
    parser.add_argument('-c', '--ncpu', metavar='NUMBER', required=False, default=1,
                        help='number of cpus used for computation. Default: 1.')
    parser.add_argument('--store_comp_id', action='store_true', default=False,
                        help='store compound id in output (only for debug).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "input": input_fname = v
        if o == "out": output_fname = v
        if o == "verbose": verbose = v
        if o == "radius": radius = int(v)
        if o == "keep_stereo": keep_stereo = v
        if o == "ncpu": ncpu = int(v)
        if o == "max_heavy_atoms": max_heavy_atoms = int(v)
        if o == "keep_mols": keep_mols = v
        if o == "store_comp_id": store_comp_id = v

    main(input_fname=input_fname,
         output_fname=output_fname,
         keep_mols=keep_mols,
         radius=radius,
         keep_stereo=keep_stereo,
         max_heavy_atoms=max_heavy_atoms,
         ncpu=ncpu,
         store_comp_id=store_comp_id,
         verbose=verbose)


if __name__ == '__main__':
    entry_point()
