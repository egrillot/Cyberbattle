"""This file provides the requested data source that the SOC analyst can observe on a machine."""


from typing import List
import numpy as np
from ...utils.markov_models import MarkovProcess

class Data_source(MarkovProcess):
    """Data_source class."""

    def __init__(self, data_source: str, description: str, states: List[str], initial_distribution: np.ndarray, transition_matrix: np.ndarray) -> None:
        """Init data_source, description and the markov process."""            
        self.data_source = data_source
        self.description = description
        
        super().__init__(states, initial_distribution, transition_matrix)

    def get_data_source(self) -> str:
        """Return the data_source name."""
        return self.data_source
    
    def call(self) -> str:
        """Return the data source triggered."""
        data_source = super().call()

        return f"{self.data_source}: {data_source}"

    def get_description(self) -> str:
        """Return the data source description."""
        return self.description
    
    def get_actions(self) -> List[str]:
        """Return the different data sources able to be triggered."""
        return ['{}: {}'.format(self.data_source, action) for action in self.get_states()]


class Quiet(Data_source):
    """No activity"""

    def __init__(self) -> None:
        """Init the Data_source Quiet."""
        data_source = 'Quiet'
        description = 'The user is just doing nothing'
        states = ['No activity']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class ActiveDirectory(Data_source):
    """ActiveDirectory class."""

    def __init__(self) -> None:
        """Init the Data_source ActiveDirectory."""
        data_source = 'Active Directory'
        description = 'A database and set of services that allows administrators to manage permissions, access to network resources, and stored data objects (user, group, application, or devices)'
        states=[
            'Active Directory Credential Request',
            'Active Directory Object Access',
            'Active Directory Object Creation',
            'Active Directory Object Deletion',
            'Active Directory Object Modification'
        ]
        initial_distribution=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        transition_matrix = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class ApplicationLog(Data_source):
    """ApplicationLog class."""

    def __init__(self) -> None:
        """Init the Data_source ApplicationLog."""
        data_source = 'Application Log'
        description = 'Events collected by third-party services such as mail servers, web applications, or other appliances (not by the native OS or platform)'
        states = ['Application Log Content']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Certificate(Data_source):
    """Certificate class."""

    def __init__(self) -> None:
        """Init the Data_source Certificate."""
        data_source = 'Certificate'
        description = "A digital document, which highlights information such as the owner's identity, used to instill trust in public keys used while encrypting network communications"
        states = ['Certificate']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class CloudService(Data_source):
    """CloudService class."""

    def __init__(self) -> None:
        """Init the Data_source CloudService."""
        data_source = 'Cloud Service'
        description = 'Infrastructure, platforms, or software that are hosted on-premise or by third-party providers, made available to users through network connections and/or APIs'
        states = [
                'Cloud Service Disable',
                'Cloud Service Enumeration',
                'Cloud Service Metadata',
                'Cloud Service Modification'
            ]
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25]
            ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class CloudStorage(Data_source):
    """CloudStorage class."""

    def __init__(self) -> None:
        """Init the Data_source CloudStorage."""
        data_source = 'Cloud Storage'
        description = 'Data object storage infrastructure hosted on-premise or by third-party providers, made available to users through network connections and/or APIs'
        states = [
                'Cloud Storage Access',
                'Cloud Storage Creation',
                'Cloud Storage Enumeration',
                'Cloud Storage Metadata',
                'Cloud Storage Modification'
            ]
        initial_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        transition_matrix = np.array([
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2]
            ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Cluster(Data_source):
    """Cluster class."""

    def __init__(self) -> None:
        """Init the Data_source Cluster."""
        data_source = 'Cluster'
        description = 'A set of containerized computing resources that are managed together but have separate nodes to execute various tasks and/or applications'
        states = ['Cluster Metadata']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Command(Data_source):
    """Command class."""

    def __init__(self) -> None:
        """Init the Data_source Command."""
        data_source = 'Command'
        description = 'A directive given to a computer program, acting as an interpreter of some kind, in order to perform a specific task'
        states = ['Command Execution']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Container(Data_source):
    """Container class."""

    def __init__(self) -> None:
        """Init the Data_source Container."""
        data_source = 'Container'
        description = 'A standard unit of virtualized software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another'
        states = [
            'Container Creation',
            'Container Enumeration',
            'Container Metadata',
            'Container Start'
        ]
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25]
                ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class DomainName(Data_source):
    """DomainName class."""

    def __init__(self) -> None:
        """Init the Data_source DomainName."""
        data_source = 'Domain Name'
        description = 'Information obtained (commonly through registration or activity logs) regarding one or more IP addresses registered with human readable names (ex: mitre.org)'
        states = ['Active DNS', 'Domain Registration', 'Passive DNS'],
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
            ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Drive(Data_source):
    """Drive class."""

    def __init__(self) -> None:
        """Init the Data_source Drive."""
        data_source = 'Drive'
        description = 'A non-volatile data storage device (hard drive, floppy disk, USB flash drive) with at least one formatted partition, typically mounted to the file system and/or assigned a drive letter'
        states = ['Drive Access', 'Drive Creation', 'Drive Modifcation']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
            ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Driver(Data_source):
    """Driver class."""

    def __init__(self) -> None:
        """Init the Data_source Driver."""
        data_source = 'Driver'
        description = 'A computer program that operates or controls a particular type of device that is attached to a computer. Provides a software interface to hardware devices, enabling operating systems and other computer programs to access hardware functions without needing to know precise details about the hardware being used'
        states = ['Driver Load', 'Driver Metadata']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
            ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class File(Data_source):
    """File class."""

    def __init__(self) -> None:
        """Init the Data_source File."""
        data_source = 'File'
        description = 'A computer resource object, managed by the I/O system, for storing data (such as images, text, videos, computer programs, or any wide variety of other media)'
        states = ['File Accesss', 'File Creation', 'File Deletion', 'File Metadata', 'File Modification']
        initial_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        transition_matrix = np.array([
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2]
            ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class FirewallManage(Data_source):
    """FirewallManage class."""

    def __init__(self) -> None:
        """Init the Data_source Firewall."""
        data_source = 'Firewall'
        description = 'A network security system, running locally on an endpoint or remotely as a service (ex: cloud environment), that monitors and controls incoming/outgoing network traffic based on predefined rules'
        states = ['Firewall Disable', 'Firewall Enumeration', 'Firewall Metadata', 'Firewall Rule Modification'],
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Firmware(Data_source):
    """Firmware class."""

    def __init__(self) -> None:
        """Init the Data_source Firmware."""
        data_source = 'Firmware'
        description = 'Computer software that provides low-level control for the hardware and device(s) of a host, such as BIOS or UEFI/EFI'
        states = ['Firmware Modification']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Group(Data_source):
    """Group class."""

    def __init__(self) -> None:
        """Init the Data_source Group."""
        data_source = 'Group'
        description = 'A collection of multiple user accounts that share the same access rights to the computer and/or network resources and have common security rights'
        states = ['Group Enumeration', 'Group Metadata', 'Group Modification']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
            ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Image(Data_source):
    """Image class."""

    def __init__(self) -> None:
        """Init the Data_source Image."""
        data_source = 'Image'
        description = 'A single file used to deploy a virtual machine/bootable disk into an on-premise or third-party cloud environment'
        states = ['Image Creation', 'Image Deletion', 'Image Metadata', 'Image Modification']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Instance(Data_source):
    """Instance class."""

    def __init__(self) -> None:
        """Init the Data_source Instance."""
        data_source = 'Instance'
        description = 'A virtual server environment which runs workloads, hosted on-premise or by third-party cloud providers'
        states = [
                'Instance Creation',
                'Instance Deletion',
                'Instance Enumeration',
                'Instance Metadata', 'Instance Metadata',
                'Instance Modification',
                'Instance Start',
                'Instance Stop'
                ]
        initial_distribution = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
        transition_matrix = np.array([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]
            ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class InternetScan(Data_source):
    """InternetScan class."""

    def __init__(self) -> None:
        """Init the Data_source InternetScan."""
        data_source = 'Internet Scan'
        description = 'Information obtained (commonly via active network traffic probes or web crawling) regarding various types of resources and servers connected to the public Internet'
        states = ['Response Content', 'Response Metadata']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Kernel(Data_source):
    """Kernel class."""

    def __init__(self) -> None:
        """Init the Data_source Kernel."""
        data_source = 'Kernel'
        description = 'A computer program, at the core of a computer OS, that resides in memory and facilitates interactions between hardware and software components'
        states = ['Kernel Module Load']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class LogonSession(Data_source):
    """LogonSession class."""

    def __init__(self) -> None:
        """Init the Data_source LogonSession."""
        data_source = 'Logon Session'
        description = 'Logon occurring on a system or resource (local, domain, or cloud) to which a user/device is gaining access after successful authentication and authorizaton'
        states = ['Logon Session Creation', 'Logon Session Metadata']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class MalwareRepository(Data_source):
    """MalwareRepository class."""

    def __init__(self) -> None:
        """Init the Data_source MalwareRepository."""
        data_source = 'Malware Repository'
        description = 'Information obtained (via shared or submitted samples) regarding malicious software (droppers, backdoors, etc.) used by adversaries'
        states = ['Malware Repository Creation', 'Malware Repository Metadata', 'Passive DNS']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Module(Data_source):
    """Module class."""

    def __init__(self) -> None:
        """Init the Data_source Module."""
        data_source = 'Module'
        description = 'Executable files consisting of one or more shared classes and interfaces, such as portable executable (PE) format binaries/dynamic link libraries (DLL), executable and linkable format (ELF) binaries/shared libraries, and Mach-O format binaries/shared libraries'
        states = ['Module Load']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class NamedPipe(Data_source):
    """NamedPipe class."""

    def __init__(self) -> None:
        """Init the Data_source NamedPipe."""
        data_source = 'Named Pipe'
        description = 'Mechanisms that allow inter-process communication locally or over the network. A named pipe is usually found as a file and processes attach to it'
        states = ['Named Pipe Metadata']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class NetworkShare(Data_source):
    """NetworkShare class."""

    def __init__(self) -> None:
        """Init the Data_source NetworkShare."""
        data_source = 'Network Share'
        description = 'A storage resource (typically a folder or drive) made available from one host to others using network protocols, such as Server Message Block (SMB) or Network File System (NFS)'
        states = ['Network Share Access']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class NetworkTraffic(Data_source):
    """NetworkTraffic class."""

    def __init__(self) -> None:
        """Init the Data_source NetworkTraffic."""
        data_source = 'Network Traffic'
        description = 'Data transmitted across a network (ex: Web, DNS, Mail, File, etc.), that is either summarized (ex: Netflow) and/or captured as raw data in an analyzable format (ex: PCAP)'
        states = ['Network Connection Creation', 'Network Traffic Content', 'Network Traffic Flow']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Persona(Data_source):
    """Persona class."""

    def __init__(self) -> None:
        """Init the Data_source Persona."""
        data_source = 'Persona'
        description = 'A malicious online profile representing a user commonly used by adversaries to social engineer or otherwise target victims'
        states = ['Social Media']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Pod(Data_source):
    """Pod class."""

    def __init__(self) -> None:
        """Init the Data_source Pod."""
        data_source = 'Pod'
        description = 'A single unit of shared resources within a cluster, comprised of one or more containers'
        states = ['Pod Creation', 'Pod Enumeration', 'Pod Metadata', 'Pod Modification']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Process(Data_source):
    """Process class."""

    def __init__(self) -> None:
        """Init the Data_source Process."""
        data_source = 'Process'
        description = 'Instances of computer programs that are being executed by at least one thread. Processes have memory space for process executables, loaded modules (DLLs or shared libraries), and allocated memory regions containing everything from user input to application-specific data structures'
        states = ['OS API Execution', 'Process Access', 'Process Creation', 'Process Metadata', 'Process Modification', 'Process Termination']
        initial_distribution = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
        transition_matrix = np.array([
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.2, 0.1, 0.1, 0.2, 0.2, 0.2]
            ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class ScheduledJob(Data_source):
    """ScheduledJob class."""

    def __init__(self) -> None:
        """Init the Data_source ScheduledJob."""
        data_source = 'Scheduled Job'
        description = 'Automated tasks that can be executed at a specific time or on a recurring schedule running in the background (ex: Cron daemon, task scheduler, BITS)'
        states = ['Scheduled Job Creation', 'Scheduled Job Metadata', 'Scheduled Job Modification']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Script(Data_source):
    """Script class."""

    def __init__(self) -> None:
        """Init the Data_source Script."""
        data_source = 'Script'
        description = 'A file or stream containing a list of commands, allowing them to be launched in sequence'
        states = ['Script Execution']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class SensorHealth(Data_source):
    """SensorHealth class."""

    def __init__(self) -> None:
        """Init the Data_source SensorHealth."""
        data_source = 'Sensor Health'
        description = 'Information from host telemetry providing insights about system status, errors, or other notable functional activity'
        states = ['Host Status']
        initial_distribution = np.array([1.])
        transition_matrix = np.array([[1.]])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Service(Data_source):
    """Service class."""

    def __init__(self) -> None:
        """Init the Data_source Service."""
        data_source = 'Service'
        description = 'A computer process that is configured to execute continuously in the background and perform system tasks, in some cases before any user has logged in'
        states = ['Service Creation', 'Service Metadata', 'Service Modification']
        initial_distribution = np.array([0.3, 0.3, 0.4])
        transition_matrix = np.array([
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.4]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Snapshot(Data_source):
    """Snapshot class."""

    def __init__(self) -> None:
        """Init the Data_source Snapshot."""
        data_source = 'Snapshot'
        description = 'A point-in-time copy of cloud volumes (files, settings, etc.) that can be created and/or deployed in cloud environments'
        states = ['Snapshot Creation', 'Snapshot Deletion', 'Snapshot Enumeration', 'Snapshot Metadata', 'Snapshot Modification']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
            [0.12, 0.22, 0.22, 0.22, 0.22],
            [0.12, 0.22, 0.22, 0.22, 0.22],
            [0.12, 0.22, 0.22, 0.22, 0.22],
            [0.12, 0.22, 0.22, 0.22, 0.22],
            [0.12, 0.22, 0.22, 0.22, 0.22]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class UserAccount(Data_source):
    """UserAccount class."""

    def __init__(self) -> None:
        """Init the Data_source UserAccount."""
        data_source = 'User Account'
        description = 'A profile representing a user, device, service, or application used to authenticate and access resources'
        states = ['User Account Authentification', 'User Account Creation', 'User Account Deletion', 'User Account Metadata', 'User Account Modification']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class Volume(Data_source):
    """Volume class."""

    def __init__(self) -> None:
        """Init the Data_source Volume."""
        data_source = 'Volume'
        description = 'Block object storage hosted on-premise or by third-party providers, typically made available to resources as virtualized hard drives'
        states = ['Volume Creation', 'Volume Deletion', 'Volume Enumeration', 'Volume Metadata', 'Volume Modification']
        initial_distribution = np.array([0.12, 0.22, 0.22, 0.22, 0.22])
        transition_matrix = np.array([
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22],
                [0.12, 0.22, 0.22, 0.22, 0.22]
            ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class WebCredential(Data_source):
    """WebCredential class."""

    def __init__(self) -> None:
        """Init the Data_source WebCredential."""
        data_source = 'Web Credential'
        description = 'Credential material, such as session cookies or tokens, used to authenticate to web applications and services'
        states = ['Web Credential Creation', 'Web Credential Usage']
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        
        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class WindowsRegistry(Data_source):
    """WindowsRegistry class."""

    def __init__(self) -> None:
        """Init the Data_source WindowsRegistry."""
        data_source = 'Windows Registry'
        description = 'A Windows OS hierarchical database that stores much of the information and settings for software programs, hardware devices, user preferences, and operating-system configurations'
        states = ['Windows Registry key Access', 'Windows Registry key Creation', 'Windows Registry key Deletion', 'Windows Registry key Modification']
        initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
        transition_matrix = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)


class WMI(Data_source):
    """WMI class."""

    def __init__(self) -> None:
        """Init the Data_source WMI."""
        data_source = 'WMI'
        description = 'The infrastructure for management data and operations that enables local and remote management of Windows personal computers and servers'
        states = ['WMI Creation']
        initial_distribution = np.array([[1.]])
        transition_matrix = np.array([[1.]])

        super().__init__(data_source, description, states, initial_distribution, transition_matrix)
