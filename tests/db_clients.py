# OPTIMADE, MPContribs databases

!pip install optimade[http_client]
!pip install mpcontribs-client

from optimade.client import OptimadeClient

client = OptimadeClient(
    include_providers={"mp"},
)
# client.get('elements HAS "Bi"')
client.list_properties("structures")

from mpcontribs.client import Client

client = Client()
client.available_query_params()  # print list of available query parameters

query = {"formula__contains": "Bi", "project": "qmof"}
fields = ["id", "identifier", "formula", "structures"]
data = client.query_contributions(
    query=query, fields=fields, sort="", paginate=False
)



from optimade.client import OptimadeClient
from optimade.adapters import Structure as OptimadeStructure


def get_structure(source, material_id):
    """
    Fetches a structure from an OPTIMADE provider, saves it as a CIF, 
    and returns the absolute file path.
    source : str
    	The database alias ('mp', 'cod', 'nomad', 'oqmd', 'aflow') or a full valid OPTIMADE URL.
    material_id : str
    	The ID of the material (e.g., 'mp-23152' or '1010068').
    Returns 
    structure_path : str
    	Path to the saved structure.
    """
    provider_urls = {
        "mp": "https://optimade.materialsproject.org",
        "cod": "https://www.crystallography.net/cod/optimade",
        "oqmd": "http://oqmd.org/optimade/",
        "nomad": "https://nomad-lab.eu/prod/rae/optimade/",
        "aflow": "http://aflow.org/API/optimade/"}
    base_url = provider_urls.get(source, source)
    client = OptimadeClient(base_urls=[base_url])
    print(f"Querying {base_url} for {material_id}...")
    filter_str = f'id="{material_id}"'
    client.get(filter=filter_str)
    results = client.all_results["structures"][filter_str][base_url]["data"]
    entry = results[0]
    pmg_structure = OptimadeStructure(entry).as_pymatgen
    structure_path = os.path.abspath(f"{material_id}.cif")
    pmg_structure.to(filename=structure_path)
    print(f"Structure saved to: {structure_path}")
    
    return structure_path
