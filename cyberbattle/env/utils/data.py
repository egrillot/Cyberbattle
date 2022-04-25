"""This file provides the requested data source that the SOC analyst can observe on a machine."""

from copy import deepcopy
from typing import List
import numpy as np
from ...utils.functions import kahansum

class Data_source:
    """Data_source class."""

    def __init__(self, data_source: str, description: str, actions: List[str], initial_distribution: np.ndarray, transition_matrix: np.ndarray, last_call: str='Stop') -> None:
        """Init data_source, description and the markov process."""
        if 'Stop' not in actions:
            actions.append('Stop')
        
        action_count = len(actions)
        initial_distribution_shape = initial_distribution.shape
        transition_matrix_shapes = transition_matrix.shape

        if len(initial_distribution_shape) != 1:
            raise ValueError('The provided initial distribution has shape of length {} instead of shape of length 1.'.format(len(initial_distribution_shape)))
        
        if initial_distribution_shape[0] != action_count - 1:
            raise ValueError('The provided initial distribution is a {} dimensional vector but it requires a {} dimensional vector'.format(initial_distribution_shape[0], action_count-1))

        if kahansum(initial_distribution) != 1:
            raise ValueError("The sum over the initial distribution : {} isn't equal to 1.".format(initial_distribution))
        
        if len(transition_matrix_shapes) != 2:
            raise ValueError('The provided transition matrix has shape of length {} instead of shape of length 2.'.format(len(transition_matrix_shapes)))
        
        if transition_matrix_shapes != (action_count, action_count):
            raise ValueError('The provided transition matrix is a {} dimensional vector but it requires a {} dimensional vector.'.format(transition_matrix_shapes, (action_count, action_count)))

        for i in range(action_count):

            if kahansum(transition_matrix[i, :]) != 1:
                raise ValueError("The sum over the row {} in the transition matrix isn't equal to 1".format(i))
            
        self.data_source = data_source
        self.description = description
        self.actions = deepcopy(actions)
        actions.remove('Stop')
        self.initial_actions = actions
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.last_call = last_call

    def get_data_source(self) -> str:
        """Return the data_source name."""
        return self.data_source

    def get_description(self) -> str:
        """Return the data source description."""
        return self.description
    
    def call(self) -> str:
        """Return an action that use the data source with respect to probability distributions and the last event."""
        if self.last_call == 'Stop':
            action = np.random.choice(self.initial_actions, p=self.initial_distribution)
        
        else:
            action = np.random.choice(self.actions, p=self.transition_matrix[self.actions.index(self.last_call), :])
        
        self.last_call = action

        return '{}: {}'.format(self.data_source, action) if action != 'Stop' else 'Stop'
    
    def get_hash_encoding(self) -> int:
        """Return the hash encoding of the data source name."""
        return hash(self.data_source)

    def reset(self) -> None:
        """Reset the current last call."""
        self.last_call = 'Stop'
    
    def get_actions(self) -> List[str]:
        """Return the actions applicable to the data source."""
        return ['{}: {}'.format(self.data_source, action) for action in self.initial_actions if action != 'Stop'] + ['Stop']

class ActiveDirectory(Data_source):
    """ActiveDirectory class."""

    def __init__(self) -> None:
        """Init the Data_source ActiveDirectory."""
        data_source = 'Active Directory'
        description = 'A database and set of services that allows administrators to manage permissions, access to network resources, and stored data objects (user, group, application, or devices)'
        actions = ['Active Directory Credential Request', 'Active Directory Object Access', 'Active Directory Object Creation', 'Active Directory Object Deletion', 'Active Directory Object Modification', 'Stop']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.02, 0.2, 0.09, 0.09, 0.2, 0.4],
            [0.02, 0.08, 0.1, 0.1, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class ApplicationLog(Data_source):
    """ApplicationLog class."""

    def __init__(self) -> None:
        """Init the Data_source ApplicationLog."""
        data_source = 'Application Log'
        description = 'Events collected by third-party services such as mail servers, web applications, or other appliances (not by the native OS or platform)'
        actions = ['Application Log Content', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])

        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Certificate(Data_source):
    """Certificate class."""

    def __init__(self) -> None:
        """Init the Data_source Certificate."""
        data_source = 'Certificate'
        description = "A digital document, which highlights information such as the owner's identity, used to instill trust in public keys used while encrypting network communications"
        actions = ['Certificate Registration', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class CloudService(Data_source):
    """CloudService class."""

    def __init__(self) -> None:
        """Init the Data_source CloudService."""
        data_source = 'Cloud Service'
        description = 'Infrastructure, platforms, or software that are hosted on-premise or by third-party providers, made available to users through network connections and/or APIs'
        actions = ['Cloud Service Disable', 'Cloud Service Enumeration', 'Cloud Service Metadata', 'Cloud Service Modification', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class CloudStorage(Data_source):
    """CloudStorage class."""

    def __init__(self) -> None:
        """Init the Data_source CloudStorage."""
        data_source = 'Cloud Storage'
        description = 'Data object storage infrastructure hosted on-premise or by third-party providers, made available to users through network connections and/or APIs'
        actions = ['Cloud Storage Access', 'Cloud Storage Creation', 'Cloud Storage Enumeration', 'Cloud Storage Metadata', 'Cloud Storage Modification', 'Stop']
        initial_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        transition_matrix = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Cluster(Data_source):
    """Cluster class."""

    def __init__(self) -> None:
        """Init the Data_source Cluster."""
        data_source = 'Cluster'
        description = 'A set of containerized computing resources that are managed together but have separate nodes to execute various tasks and/or applications'
        actions = ['Cluster Metadata', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Command(Data_source):
    """Command class."""

    def __init__(self) -> None:
        """Init the Data_source Command."""
        data_source = 'Command'
        description = 'A directive given to a computer program, acting as an interpreter of some kind, in order to perform a specific task'
        actions = ['Command Execution', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Container(Data_source):
    """Container class."""

    def __init__(self) -> None:
        """Init the Data_source Container."""
        data_source = 'Container'
        description = 'A standard unit of virtualized software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another'
        actions = ['Container Creation', 'Container Enumeration', 'Container Metadata', 'Container Start', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class DomainName(Data_source):
    """DomainName class."""

    def __init__(self) -> None:
        """Init the Data_source DomainName."""
        data_source = 'Domain Name'
        description = 'Information obtained (commonly through registration or activity logs) regarding one or more IP addresses registered with human readable names (ex: mitre.org)'
        actions = ['Active DNS', 'Domain Registration', 'Passive DNS', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Drive(Data_source):
    """Drive class."""

    def __init__(self) -> None:
        """Init the Data_source Drive."""
        data_source = 'Drive'
        description = 'A non-volatile data storage device (hard drive, floppy disk, USB flash drive) with at least one formatted partition, typically mounted to the file system and/or assigned a drive letter'
        actions = ['Drive Access', 'Drive Creation', 'Drive Modifcation', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Driver(Data_source):
    """Driver class."""

    def __init__(self) -> None:
        """Init the Data_source Driver."""
        data_source = 'Driver'
        description = 'A computer program that operates or controls a particular type of device that is attached to a computer. Provides a software interface to hardware devices, enabling operating systems and other computer programs to access hardware functions without needing to know precise details about the hardware being used'
        actions = ['Driver Load', 'Driver Metadata', 'Stop']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class File(Data_source):
    """File class."""

    def __init__(self) -> None:
        """Init the Data_source File."""
        data_source = 'File'
        description = 'A computer resource object, managed by the I/O system, for storing data (such as images, text, videos, computer programs, or any wide variety of other media)'
        actions = ['File Accesss', 'File Creation', 'File Deletion', 'File Metadata', 'File Modification', 'Stop']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.02, 0.2, 0.09, 0.09, 0.2, 0.4],
            [0.02, 0.08, 0.1, 0.1, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Firewall(Data_source):
    """Firewall class."""

    def __init__(self) -> None:
        """Init the Data_source Firewall."""
        data_source = 'Firewall'
        description = 'A network security system, running locally on an endpoint or remotely as a service (ex: cloud environment), that monitors and controls incoming/outgoing network traffic based on predefined rules'
        actions = ['Firewall Disable', 'Firewall Enumeration', 'Firewall Metadata', 'Firewall Rule Modification', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Firmware(Data_source):
    """Firmware class."""

    def __init__(self) -> None:
        """Init the Data_source Firmware."""
        data_source = 'Firmware'
        description = 'Computer software that provides low-level control for the hardware and device(s) of a host, such as BIOS or UEFI/EFI'
        actions = ['Firmware Modification', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Group(Data_source):
    """Group class."""

    def __init__(self) -> None:
        """Init the Data_source Group."""
        data_source = 'Group'
        description = 'A collection of multiple user accounts that share the same access rights to the computer and/or network resources and have common security rights'
        actions = ['Group Enumeration', 'Group Metadata', 'Group Modification', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Image(Data_source):
    """Image class."""

    def __init__(self) -> None:
        """Init the Data_source Image."""
        data_source = 'Image'
        description = 'A single file used to deploy a virtual machine/bootable disk into an on-premise or third-party cloud environment'
        actions = ['Image Creation', 'Image Deletion', 'Image Metadata', 'Image Modification', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Instance(Data_source):
    """Instance class."""

    def __init__(self) -> None:
        """Init the Data_source Instance."""
        data_source = 'Instance'
        description = 'A virtual server environment which runs workloads, hosted on-premise or by third-party cloud providers'
        actions = ['Instance Creation', 'Instance Deletion', 'Instance Enumeration', 'Instance Metadata', 'Instance Metadata', 'Instance Modification', 'Instance Start', 'Instance Stop', 'Stop']
        initial_distribution = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
        transition_matrix = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class InternetScan(Data_source):
    """InternetScan class."""

    def __init__(self) -> None:
        """Init the Data_source InternetScan."""
        data_source = 'Internet Scan'
        description = 'Information obtained (commonly via active network traffic probes or web crawling) regarding various types of resources and servers connected to the public Internet'
        actions = ['Response Content', 'Response Metadata', 'Stop']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Kernel(Data_source):
    """Kernel class."""

    def __init__(self) -> None:
        """Init the Data_source Kernel."""
        data_source = 'Kernel'
        description = 'A computer program, at the core of a computer OS, that resides in memory and facilitates interactions between hardware and software components'
        actions = ['Kernel Module Load', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class LogonSession(Data_source):
    """LogonSession class."""

    def __init__(self) -> None:
        """Init the Data_source LogonSession."""
        data_source = 'Logon Session'
        description = 'Logon occurring on a system or resource (local, domain, or cloud) to which a user/device is gaining access after successful authentication and authorizaton'
        actions = ['Logon Session Creation', 'Logon Session Metadata', 'Stop']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class MalwareRepository(Data_source):
    """MalwareRepository class."""

    def __init__(self) -> None:
        """Init the Data_source MalwareRepository."""
        data_source = 'Malware Repository'
        description = 'Information obtained (via shared or submitted samples) regarding malicious software (droppers, backdoors, etc.) used by adversaries'
        actions = ['Malware Repository Creation', 'Malware Repository Metadata', 'Stop']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Module(Data_source):
    """Module class."""

    def __init__(self) -> None:
        """Init the Data_source Module."""
        data_source = 'Module'
        description = 'Executable files consisting of one or more shared classes and interfaces, such as portable executable (PE) format binaries/dynamic link libraries (DLL), executable and linkable format (ELF) binaries/shared libraries, and Mach-O format binaries/shared libraries'
        actions = ['Module Load', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class NamedPipe(Data_source):
    """NamedPipe class."""

    def __init__(self) -> None:
        """Init the Data_source NamedPipe."""
        data_source = 'Named Pipe'
        description = 'Mechanisms that allow inter-process communication locally or over the network. A named pipe is usually found as a file and processes attach to it'
        actions = ['Named Pipe Metadata', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class NetworkShare(Data_source):
    """NetworkShare class."""

    def __init__(self) -> None:
        """Init the Data_source NetworkShare."""
        data_source = 'Network Share'
        description = 'A storage resource (typically a folder or drive) made available from one host to others using network protocols, such as Server Message Block (SMB) or Network File System (NFS)'
        actions = ['Network Share Access', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class NetworkTraffic(Data_source):
    """NetworkTraffic class."""

    def __init__(self) -> None:
        """Init the Data_source NetworkTraffic."""
        data_source = 'Network Traffic'
        description = 'Data transmitted across a network (ex: Web, DNS, Mail, File, etc.), that is either summarized (ex: Netflow) and/or captured as raw data in an analyzable format (ex: PCAP)'
        actions = ['Network Connection Creation', 'Network Traffic Content', 'Network Traffic Flow', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Persona(Data_source):
    """Persona class."""

    def __init__(self) -> None:
        """Init the Data_source Persona."""
        data_source = 'Persona'
        description = 'A malicious online profile representing a user commonly used by adversaries to social engineer or otherwise target victims'
        actions = ['Social Media', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Pod(Data_source):
    """Pod class."""

    def __init__(self) -> None:
        """Init the Data_source Pod."""
        data_source = 'Pod'
        description = 'A single unit of shared resources within a cluster, comprised of one or more containers'
        actions = ['Pod Creation', 'Pod Enumeration', 'Pod Metadata', 'Pod Modification', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Process(Data_source):
    """Process class."""

    def __init__(self) -> None:
        """Init the Data_source Process."""
        data_source = 'Process'
        description = 'Instances of computer programs that are being executed by at least one thread. Processes have memory space for process executables, loaded modules (DLLs or shared libraries), and allocated memory regions containing everything from user input to application-specific data structures'
        actions = ['OS API Execution', 'Process Access', 'Process Creation', 'Process Metadata', 'Process Modification', 'Process Termination', 'Stop']
        initial_distribution = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
        transition_matrix = np.array([
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class ScheduledJob(Data_source):
    """ScheduledJob class."""

    def __init__(self) -> None:
        """Init the Data_source ScheduledJob."""
        data_source = 'Scheduled Job'
        description = 'Automated tasks that can be executed at a specific time or on a recurring schedule running in the background (ex: Cron daemon, task scheduler, BITS)'
        actions = ['Scheduled Job Creation', 'Scheduled Job Metadata', 'Scheduled Job Modification', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Script(Data_source):
    """Script class."""

    def __init__(self) -> None:
        """Init the Data_source Script."""
        data_source = 'Script'
        description = 'A file or stream containing a list of commands, allowing them to be launched in sequence'
        actions = ['Script Execution', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class SensorHealth(Data_source):
    """SensorHealth class."""

    def __init__(self) -> None:
        """Init the Data_source SensorHealth."""
        data_source = 'Sensor Health'
        description = 'Information from host telemetry providing insights about system status, errors, or other notable functional activity'
        actions = ['Host Status', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Service(Data_source):
    """Service class."""

    def __init__(self) -> None:
        """Init the Data_source Service."""
        data_source = 'Service'
        description = 'A computer process that is configured to execute continuously in the background and perform system tasks, in some cases before any user has logged in'
        actions = ['Service Creation', 'Service Metadata', 'Service Modification', 'Stop']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Snapshot(Data_source):
    """Snapshot class."""

    def __init__(self) -> None:
        """Init the Data_source Snapshot."""
        data_source = 'Snapshot'
        description = 'A point-in-time copy of cloud volumes (files, settings, etc.) that can be created and/or deployed in cloud environments'
        actions = ['Snapshot Creation', 'Snapshot Deletion', 'Snapshot Enumeration', 'Snapshot Metadata', 'Snapshot Modification', 'Stop']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.02, 0.2, 0.09, 0.09, 0.2, 0.4],
            [0.02, 0.08, 0.1, 0.1, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class UserAccount(Data_source):
    """UserAccount class."""

    def __init__(self) -> None:
        """Init the Data_source UserAccount."""
        data_source = 'User Account'
        description = 'A profile representing a user, device, service, or application used to authenticate and access resources'
        actions = ['User Account Authentification', 'User Account Creation', 'User Account Deletion', 'User Account Metadata', 'User Account Modification', 'Stop']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.02, 0.2, 0.09, 0.09, 0.2, 0.4],
            [0.02, 0.08, 0.1, 0.1, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class Volume(Data_source):
    """Volume class."""

    def __init__(self) -> None:
        """Init the Data_source Volume."""
        data_source = 'Volume'
        description = 'Block object storage hosted on-premise or by third-party providers, typically made available to resources as virtualized hard drives'
        actions = ['Volume Creation', 'Volume Deletion', 'Volume Enumeration', 'Volume Metadata', 'Volume Modification', 'Stop']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.02, 0.2, 0.09, 0.09, 0.2, 0.4],
            [0.02, 0.08, 0.1, 0.1, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.04, 0.04, 0.02, 0.2, 0.2, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class WebCredential(Data_source):
    """WebCredential class."""

    def __init__(self) -> None:
        """Init the Data_source WebCredential."""
        data_source = 'Web Credential'
        description = 'Credential material, such as session cookies or tokens, used to authenticate to web applications and services'
        actions = ['Web Credential Creation', 'Web Credential Usage', 'Stop']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class WindowsRegistry(Data_source):
    """WindowsRegistry class."""

    def __init__(self) -> None:
        """Init the Data_source WindowsRegistry."""
        data_source = 'Windows Registry'
        description = 'A Windows OS hierarchical database that stores much of the information and settings for software programs, hardware devices, user preferences, and operating-system configurations'
        actions = ['Windows Registry key Access', 'Windows Registry key Creation', 'Windows Registry key Deletion', 'Windows Registry key Modification', 'Stop']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)


class WMI(Data_source):
    """WMI class."""

    def __init__(self) -> None:
        """Init the Data_source WMI."""
        data_source = 'WMI'
        description = 'The infrastructure for management data and operations that enables local and remote management of Windows personal computers and servers'
        actions = ['WMI Creation', 'Stop']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        super().__init__(data_source, description, actions, initial_distribution, transition_matrix)
