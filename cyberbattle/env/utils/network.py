

import networkx as nx
from networkx import all_shortest_paths, has_path
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from PIL import Image
from .machine import Machine


class Network:
    """Defines the whole environment where agents will fight."""

    def __init__(self, machines: List[Machine], name: str):
        """Create a networkx object to define the environment and assigns for each machine the machines accessible from it.
        
        Input:
        name: environment name (str)
        machines (List[Machine]).
        """
        if len(set([m.get_instance_name() for m in machines])) != len(machines):
            raise ValueError("Machine instance names aren't unique.")

        self.name = name
        self.machine_list = machines
        self.ip_adresses_encoder = dict() # encode an ip adress to an index
        self.machine_count = len(machines)
        self.ip_to_machine = dict([(m.get_ip_adress(), m) for m in machines])
        self.instance_name_to_machine = dict([(m.get_instance_name(), m) for m in machines])
        self.graph = nx.Graph()
        added_machines = set()

        index = 0
        for machine in machines:

            ip_adress = machine.get_ip_adress()
            instance_name = machine.get_instance_name()
            im = Image.open(machine.get_url_image())
            connected_machines = machine.get_connected_machines()
            self.graph.add_node(instance_name, image=im)
            added_machines.add(instance_name)
            
            for m in connected_machines:

                if m in added_machines:

                    self.graph.add_edge(instance_name, m)

            self.ip_adresses_encoder[ip_adress] = index
            index += 1
        
        self.instance_name_to_index = dict([(m.get_instance_name(), self.ip_adresses_encoder[m.get_ip_adress()]) for m in machines])

        self.paths = np.zeros((index, index), dtype=object)

        for machine1 in self.instance_name_to_machine.keys():

            for machine2 in self.instance_name_to_machine.keys():

                if machine1 != machine2:

                    if not has_path(self.graph, source=machine1, target=machine2):

                        raise ValueError("{} and {} are not connected.".format(machine1, machine2))

                    paths = all_shortest_paths(self.graph, source=machine1, target=machine2)
                    paths = list(paths)
                    nb_path = len(paths)

                    if nb_path > 1:

                        raise ValueError('The path between 2 machines must be unique, found {} paths linking {} and {}'.format(nb_path, machine1, machine2))

                    idx1 = self.instance_name_to_index[machine1]
                    idx2 = self.instance_name_to_index[machine2]
                    self.paths[idx1, idx2] = [self.instance_name_to_machine[m] for m in paths[0]]
                    
    
    def display(self, save_figure: str=None, annotations: bool=False) -> None:
        """Display the network.
        
        Input:
        save_figure: path to save the figure, if not provided, the figure isn't saved. Defaut value: None (str)
        annotations: if True, machine instance names are displayed. Defaut value: False (bool).
        """
        pos = nx.spring_layout(self.graph, seed=1734289230)
        pos_array = np.array(list(pos.values()), dtype=float)
        max_x, max_y = np.max(pos_array, axis=0)
        bound = max(max_x, max_y) + 0.5
        pos_array = nx.rescale_layout(pos_array, scale=1 / bound ** 2)

        for key, value in zip(pos.keys(), pos_array):
            pos[key] = value + 0.5

        fig, ax = plt.subplots()

        nx.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=ax,
            arrows=True,
            arrowstyle="-",
            min_source_margin=15,
            min_target_margin=15
        )

        tr_figure = ax.transData.transform
        tr_axes = fig.transFigure.inverted().transform

        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
        icon_center = icon_size / 2.0

        for n in self.graph.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(self.graph.nodes[n]["image"])

            if annotations:
                a.annotate(n, xy=[xa - icon_center, ya - icon_center])

            a.axis("off")
        
        ax.set_title('Environment : {}'.format(self.name))
        plt.show()

        if save_figure:

            fig.savefig(save_figure)
        
    def get_paths(self) -> np.ndarray:
        """Return network paths."""
        return self.paths
    
    def get_path(self, machine1: str, machine2: str) -> List[Machine]:
        """Return the path linking both machines."""
        idx1 = self.instance_name_to_index[machine1]
        idx2 = self.instance_name_to_index[machine2]

        return self.paths[idx1, idx2]
    
    def get_machine_list(self) -> List[Machine]:
        """Return the machine list."""
        return self.machine_list
                