def spglib_structure(structure_path, symmetrize=False, symprec=1e-5):
    """
    Determine space group with spglib.
    Write dataset to file with _spglib suffix.
    Optionally overwrite structure file.
    structure_path : str
        Path to the structure file.
    symmetrize : bool
        If True, symmetrize atomic positions and lattice vectors using spglib
        and overwrite the structure file.
    symprec : float
        Symmetry precision for spglib. Default is 1e-5.
    """    
    atoms = read(structure_path)
    # Detect symmetry with spglib
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell_tuple = (lattice, positions, numbers)
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=symprec)
    if isinstance(dataset, dict):
        spacegroup = dataset["international"]
        number = dataset["number"]
    else:
        spacegroup = dataset.international
        number = dataset.number
    print(f"Detected space group: {spacegroup} ({number})")
    # Write spglib dataset to file
    structure_stem = os.path.splitext(structure_path)[0]
    spglib_path = f"{structure_stem}_spglib"
    with open(spglib_path, 'w') as f:
        f.write(str(dataset))
    # Symmetrize structure
    if symmetrize:
        standardized = spglib.standardize_cell(
            cell_tuple,
            to_primitive=True,
            no_idealize=False,
            symprec=symprec)
        std_lattice, std_positions, std_numbers = standardized
        # Map atomic numbers back to symbols
        std_symbols = [chemical_symbols[n] for n in std_numbers]
        # Create new ASE Atoms object with symmetrized geometry
        symmetrized_atoms = Atoms(
            symbols=std_symbols,
            scaled_positions=std_positions,
            cell=std_lattice,
            pbc=True)
        # Overwrite structure file
        write(structure_path, symmetrized_atoms)
        print(f"Structure symmetrized and overwritten: {structure_path}")
