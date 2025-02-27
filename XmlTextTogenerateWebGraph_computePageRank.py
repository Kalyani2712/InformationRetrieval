import networkx as nx
import matplotlib.pyplot as plt
from xml.dom.minidom import parse

# Parse XML
collection = parse("movies.xml").documentElement
if collection.hasAttribute("shelf"):
    print("Root element:", collection.getAttribute("shelf"))

# Print movie details
for movie in collection.getElementsByTagName("movie"):
    print("\n***** Movie *****")
    print("Title:", movie.getAttribute("title")) if movie.hasAttribute("title") else None
    for tag in ["type", "format", "rating", "description"]:
        print(f"{tag.capitalize()}: {movie.getElementsByTagName(tag)[0].childNodes[0].data}")

# Generate and visualize graph
def GenerateGraph():
    G = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=15)
    plt.savefig("simple_path.png")  
    plt.show()
    print("Nodes:", G.nodes(), "\nEdges:", G.edges())

GenerateGraph()
