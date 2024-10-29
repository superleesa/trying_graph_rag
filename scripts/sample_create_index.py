from trying_graph_rag.graph_rag.indexing import create_index

if __name__ == '__main__':
    documents = [
    "Dr. Emily Carter presented her groundbreaking research on renewable energy technologies at the Global Science Conference in Berlin. Her team at GreenTech Innovations has developed a new solar panel that increases efficiency by 20%.",
    "The partnership between Orion Aerospace and the Stellar Space Agency was announced yesterday. They plan to collaborate on the Artemis Mission to establish a lunar base by 2025.",
    "Chef Antonio Russo opened his new restaurant, Bella Italia, in downtown New York. The establishment is already receiving rave reviews from food critics like Sarah Lee of The Culinary Times.",
    "EcoWorld Organization awarded the Environmental Hero Award to Maya Singh for her efforts in conserving the Amazon Rainforest. The ceremony took place in São Paulo, Brazil.",
    "Tech giants InnovateX and CyberLink merged to form a new company called NextGen Solutions. The merger aims to revolutionize artificial intelligence and cybersecurity.",
    "Professor Liam O'Connor from the University of Dublin published a book titled \"The Future of Quantum Computing\". It is expected to influence tech companies like QuantumLeap Inc.",
    "The city of Tokyo will host the International Robotics Expo, featuring companies like RoboTech and AI Innovations. Experts such as Dr. Satoshi Nakamura will present their latest advancements.",
    "Musician Sofia Martinez collaborated with producer DJ Wave to release her new album \"Echoes\". The album launch event was held at the Sunset Arena in Los Angeles.",
    "Mountaineer Jack Thompson reached the summit of Mount Everest with his team from Summit Adventures. The expedition was sponsored by OutdoorGear Corp.",
    "The United Nations appointed Ambassador Aisha Hassan as the Special Envoy to the Middle East. She will work closely with leaders from Jordan, Lebanon, and Egypt to promote peace initiatives."
]   
    entity_types = ["person", "organization", "technology", "location", "event"]
    create_index(documents, entity_types)