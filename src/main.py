from rdkit import Chem


def get_cip_label(atom: Chem.Atom) -> str | None:
    if atom.HasProp("_CIPCode"):
        return str(atom.GetProp("_CIPCode"))
    return None


def get_neighbor_cip_rank(bond: Chem.Bond) -> int:
    if bond.HasProp("_CIPRank"):
        return int(bond.GetProp("_CIPRank"))
    return 9999


def find_chiral_centers(mol: Chem.Mol) -> list[tuple[int, str]]:
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return centers  # type: ignore


def custom_traverse_smiles(
    mol: Chem.Mol,
    atom_idx: int,
    visited: set[int],
    chiral_info: dict[int, str],
    is_root: bool,
    parent_idx: int | None = None,
) -> str:
    visited.add(atom_idx)
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()

    neighbors = []
    for bond in atom.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 == atom_idx and a2 != parent_idx:
            neighbors.append((a2, bond))
        elif a2 == atom_idx and a1 != parent_idx:
            neighbors.append((a1, bond))

    label = chiral_info.get(atom_idx, None)
    if label in ("R", "S"):
        neighbors = sorted(neighbors, key=lambda x: get_neighbor_cip_rank(x[1]))
        if label == "S":
            if len(neighbors) >= 3:
                neighbors[1], neighbors[2] = neighbors[2], neighbors[1]

    branches = []
    for child_idx, child_bond in neighbors:
        if child_idx not in visited:
            child_str = custom_traverse_smiles(
                mol, child_idx, visited, chiral_info, False, parent_idx=atom_idx
            )
            if is_root:
                branches.append(child_str)
            else:
                branches.append(f"({child_str})")

    return symbol + "".join(branches)


def generate_so_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    centers = find_chiral_centers(mol)
    chiral_info = {}
    for c_idx, c_label in centers:
        chiral_info[c_idx] = c_label

    visited: set[int] = set()

    result = custom_traverse_smiles(mol, 0, visited, chiral_info, True)
    return result


if __name__ == "__main__":
    smiles_in = "CC(C)[C@H]1C(=O)O[C@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)N1C"
    print("Input SMILES:", smiles_in)

    out_smiles = generate_so_smiles(smiles_in)
    print("Recursive-rule SMILES:", out_smiles)

    mol = Chem.MolFromSmiles(smiles_in)
    standard_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    print("Standard SMILES (RDKit):", standard_smiles)
