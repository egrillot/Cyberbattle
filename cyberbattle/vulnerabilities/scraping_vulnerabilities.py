# coding: utf-8
#  Script d'interopérabilité neo4j - Python CBS et automatisation
#  Création des vulns depuis le MITRE ATT&CK et implémentation automatique
#  Auteur : Matthieu TOULOUCANON pour BSSI
#  You may check out the manual for a reference on how to use this tool

from enum import IntEnum

# from taxii2client.v20 import Collection
from stix2 import Filter, MemoryStore  # , TAXIICollectionSource

from cyberbattle.simulation import model as m
from cyberbattle.simulation.model import VulnerabilityType

# LIST UTILITY


def intersect(*lists):
    """
    Returns intersection of all lists
    """
    return list(set(lists[0]).intersection(*lists[1:]))


def union(*lists):
    """
    Returns union of all lists
    """
    return list(set(lists[0]).union(*lists[1:]))


def minus(l1, l2):
    """
    Returns List1 - List2
    """
    return [i1 for i1 in l1 if i1 not in l2]


# WARNING: Some items don't have description in MITRE ATT&CK...

# connect to ATT&CK framework
# collections = {
#     "enterprise_attack": "95ecc380-afe9-11e4-9b6c-751b66dd541e",
#     "mobile_attack": "2f669986-b40b-4423-b720-4396ca6a462b",
#     "ics-attack": "02c3ef24-9cd4-48f3-a99f-b74ce24f1d34"
# }

# collection = Collection(f"https://cti-taxii.mitre.org/stix/collections/{collections['enterprise_attack']}/")
# src = TAXIICollectionSource(collection)
# print("Successfully connected to MITRE ATT&CK enterprise")

def checkForATTACKUpdate():
    """
    Checks for update of the ATT&CK json file using the CHANGELOG file (downloading the json takes too long)
    Updates automatically if the an update is available
    """
    fd = __import__('os').popen('/usr/bin/curl https://raw.githubusercontent.com/mitre/cti/master/CHANGELOG.md 2>/dev/null | md5sum')
    newver = fd.read()
    fd.close()
    oldfd = open('/root/attackversion')
    oldver = oldfd.read()
    oldfd.close()
    if newver != oldver:
        print(">>> UPDATE AVAILABLE <<<\n\nUpdating...")
        __import__('os').system('/usr/bin/curl https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json > /root/enterprise-attack.json && /usr/bin/curl https://raw.githubusercontent.com/mitre/cti/master/CHANGELOG.md 2>/dev/null | md5sum > /root/attackversion')
        print("Update finished, please restart.")
        exit()
    else:
        print("No update available")


checkForATTACKUpdate()
src = MemoryStore()
print("Loading MITRE ATT&CK file locally... (fastest way)")
src.load_from_file('/root/enterprise-attack.json')
print("Connected locally to ATT&CK enterprise")


Available_Platforms = ['Windows', 'Linux', 'AWS', 'Azure AD', 'Google Workspace', 'IaaS', 'Office 365', 'macOS', 'Containers', 'Network', 'SaaS', 'PRE']
effective_permissions_link = {
    'Remote Desktop Users': m.PrivilegeLevel.LocalUser,
    'User': m.PrivilegeLevel.LocalUser,
    'Administrator': m.PrivilegeLevel.Admin,
    'SYSTEM': m.PrivilegeLevel.System,
    'root': m.PrivilegeLevel.System
}

# GETTERS


def getPrecondition(attack) -> m.Precondition:
    """
    From an attack, gets the required permissions to perform it
    """
    perms_required = attack.get('x_mitre_permissions_required')
    # Get the minimum permission required tag 'perm' -> PrivilegeLevel -> PrivilegeEscalation.tag
    if perms_required == ['User']:  # Precondition user for mitre is nobody...
        return m.Precondition("true")
    if perms_required is not None:
        return m.Precondition(m.PrivilegeEscalation(effective_permissions_link[perms_required[0]]).tag + "".join([f"|{m.PrivilegeEscalation(effective_permissions_link[perm_req]).tag}" for perm_req in perms_required[1:]]))

    return m.Precondition("true")


def getLabels(node):
    return str(node.labels).strip(':').split(':')


def getKnownNodes(node):
    return [rel.end_node for rel in node.graph.match((node,))]


def getTacticsNames(attack):
    return [phase.phase_name for phase in attack.kill_chain_phases]


def getVulnType(attack):
    """
    Determines whether the vuln is remote or local
    """

    if 'x_mitre_remote_support' in attack.properties_populated():
        if attack.x_mitre_remote_support:
            return VulnerabilityType.REMOTE
        return VulnerabilityType.LOCAL

    if 'x_mitre_permissions_required' in attack.properties_populated():
        if max(effective_permissions_link[level] for level in attack.x_mitre_permissions_required) > m.PrivilegeLevel.LocalUser:
            return VulnerabilityType.LOCAL

    return VulnerabilityType.REMOTE


def getParentTechnique(subTechnique):
    id = subTechnique.external_references[0].external_id.split('.')[0]
    return remove_revoked_deprecated(src.query([Filter('external_references.external_id', '=', id)]))[0]


def descriptionfilter_atleastonce(attacks, labels):
    """
    Attacks so that there is at least one label in there description or name
    """
    return [attack for attack in attacks for label in labels if label.lower() in (attack.description + attack.name).lower()]


def descriptionfilter_allin(attacks, labels):
    """
    Attacks so that there are all labels in there description or name
    """
    allin = True
    ret = []
    for attack in attacks:
        for label in labels:
            if label.lower() not in (attack.description + attack.name).lower():
                allin = False
                break
        if allin:
            ret.append(attack)
        allin = True
    return ret


def remove_revoked_deprecated(stix_objects):  # Taken from mitre github
    """
    Remove any revoked or deprecated objects from queries made to the data source
    """
    # Note we use .get() because the property may not be present in the JSON data. The default is False
    # if the property is not set.
    return list(
        filter(
            lambda x: x.get("x_mitre_deprecated", False) is False and x.get("revoked", False) is False,
            stix_objects
        )
    )


def getAttacks(node):
    """
    Returns attacks (subtechniques) from a node using its labels, to be injected in getVulns for
    further parsing.
    """
    labels = getLabels(node)
    platforms = intersect(labels + ['PRE'], Available_Platforms)  # Intersect all platforms with node's ones to get platform labels
    otherlabels = minus(labels, platforms)  # All the labels except platforms'
    print("Platforms: ", platforms, " ; other labels : ", otherlabels)
    fil = [  # query subtechniques --> introspect and later query parents on demand
        Filter('type', '=', 'attack-pattern'),  # att&ck framework
        Filter('x_mitre_is_subtechnique', '=', True),  # Only subtechniques to get closer to vulns
        Filter('description', 'contains', ' ')
    ]

    # Les filtres sont des '&', donc on veut:
    # UNION sur les plateformes des query(filtre & plateforme).
    # process : query les id, faire l'union des ids et requery grâce aux ids
    # NB: union intrinsèque par le passage dans la fonction set ci-dessous, pas d'appel à la fonction union
    union_ids = list(set([q.id for platform in platforms for q in src.query(fil + [Filter('x_mitre_platforms', 'contains', platform)])]))
    print("--> Found %d unique attacks, getting them..." % len(union_ids))
    attacks_without_descfilter = [src.get(stix_id) for stix_id in union_ids]
    print("--> Got %d attacks, filtering to only keep the most relevant ones..." % len(attacks_without_descfilter))
    attacks_descfiltered = descriptionfilter_atleastonce(attacks_without_descfilter, otherlabels)
    print("--> Reduced to %d techniques by using labels :)\n" % len(attacks_descfiltered))
    return remove_revoked_deprecated(attacks_descfiltered)


def get_services(node):  # Node('Linux', 'Server', NodeID='Serveur Apache', SLA_Weight=1.0, Value=10, **{'Service;SSH': 'sysadm:1337'})
    keys = list(node)  # ['NodeID', 'Value', 'SLA_Weight', 'Service;SSH']
    services = []
    for k in keys:
        if 'Service' in k:
            service_name = k.split(";")[1]
            service_creds = node[k].split(";")
            services.append([service_name, service_creds])
    return services


def getCachedCredentials(node):
    """
    Returns an array of CachedCredentials available for a node.
    """
    return [m.CachedCredential(node=node['NodeID'], port=service[0], credential=cred) for service in get_services(node) for cred in service[1]]


class Costs(IntEnum):
    LOW = 1
    MEDIUM = 5
    HIGH = 7
    VERY_HIGH = 15


# Evaluation of costs using data sources. I tried to make it quite relevant but it can surely be improved
# TODO improve this mechanism
data_sources_costs = {
    "Application Log: Application Log Content": Costs.VERY_HIGH,
    "User Account: User Account Authentication": Costs.VERY_HIGH,
    "Command: Command Execution": Costs.MEDIUM,
    "File: File Access": Costs.LOW,
    "Network Traffic: Network Traffic Content": Costs.LOW,
    "Network Traffic: Network Traffic Flow": Costs.LOW,
    "Active Directory: Active Directory Credential Request": Costs.MEDIUM,
    "Process: Process Creation": Costs.HIGH,
    "User Account: User Account Modification": Costs.VERY_HIGH,
    'File: File Creation': Costs.MEDIUM,
    'File: File Modification': Costs.LOW,
    "Module: Module Load": Costs.MEDIUM,
    "Sensor Health: Host Status": Costs.HIGH,
    "Script: Script Execution": Costs.LOW,
    "File: File Deletion": Costs.VERY_HIGH,
    "Logon Session: Logon Session Creation": Costs.VERY_HIGH,
    "Group: Group Enumeration": Costs.VERY_HIGH,
    "Group: Group Metadata": Costs.MEDIUM,
    "File: File Metadata": Costs.LOW,
    "Kernel: Kernel Module Load": Costs.VERY_HIGH,
    "Driver: Driver Load": Costs.VERY_HIGH,
    "Process: OS API Execution": Costs.HIGH,
    "Windows Registry: Windows Registry Key Creation": Costs.MEDIUM,
    "Windows Registry: Windows Registry Key Modification": Costs.MEDIUM,
    "Active Directory: Active Directory Object Modification": Costs.VERY_HIGH,
    "Drive: Drive Modification": Costs.HIGH,
    "Network Traffic: Network Connection Creation": Costs.HIGH,
    "Image: Image Creation": Costs.HIGH,
    "Volume: Volume Enumeration": Costs.VERY_HIGH,
    "Cloud Storage: Cloud Storage Enumeration": Costs.VERY_HIGH,
    "Snapshot: Snapshot Enumeration": Costs.VERY_HIGH
}


def evaluateVulnCost(attack):
    """
    Tries to evaluate the cost of a vulnerability. Uses the data sources to do so, it's not perfect.
    """
    sources = attack.get('x_mitre_data_sources')
    cost = 0.0
    if sources is not None:
        for source in sources:
            cost += float(data_sources_costs.get(source, Costs.MEDIUM))
        if 'defense-evasion' in getTacticsNames(attack):
            return cost / 10
        return cost
    return 1.0


# TACTICS functions, should not get called explicitly
def TacticReconaissance(node, known_nodes, attack, parent_attack):
    return m.LeakedNodesId([kn['NodeID'] for kn in known_nodes])


def TacticExecution(node, known_nodes, attack, parent_attack):
    return m.PrivilegeEscalation(m.PrivilegeLevel.LocalUser)


def TacticPrivesc(node, known_nodes, attack, parent_attack):
    """
    Gets the privesc privileges result and return the associated object
    """

    if 'x_mitre_effective_permissions' in attack.properties_populated():
        levels = attack.x_mitre_effective_permissions
    elif 'x_mitre_effective_permissions' in parent_attack.properties_populated():
        levels = parent_attack.x_mitre_effective_permissions
    else:
        return m.PrivilegeEscalation(m.PrivilegeLevel.System)  # default case when no data
    return m.PrivilegeEscalation(max(effective_permissions_link[level] for level in levels))


def TacticCredAccess(node, known_nodes, attack, parent_attack):
    if len(known_nodes) == 0:
        return None
    lowestValue = min(kn['Value'] for kn in known_nodes)

    for nd in known_nodes:
        if nd['Value'] == lowestValue:
            creds = getCachedCredentials(nd)
            if len(creds) == 0:
                return None
            return m.LeakedCredentials(creds)


def TacticDiscovery(node, known_nodes, attack, parent_attack):
    return TacticReconaissance(node, known_nodes, attack, parent_attack)


def TacticCollection(node, known_nodes, attack, parent_attack):
    return TacticCredAccess(node, known_nodes, attack, parent_attack)


tactics_Outcomes = {
    'reconnaissance': TacticReconaissance,
    'execution': TacticExecution,
    'privilege-escalation': TacticPrivesc,
    'credential-access': TacticCredAccess,
    'discovery': TacticDiscovery,
    'collection': TacticCollection,
}

# Normal functions again


def getOutcome(node, attack):
    """
    Returns a complete vulnerability outcome from a node and an attack from MITRE ATT&CK
    """
    known_nodes = getKnownNodes(node)
    tacticsnames = intersect(getTacticsNames(attack), list(tactics_Outcomes))
    if tacticsnames == []:
        print("Technique \"%s\" removed (useless / irrelevent))" % attack.name)
        return None
    print("Analysing technique \"%s\" of type \"%s\" of plateform %s. More info here: %s" % (attack.name, tacticsnames[0], str(attack.x_mitre_platforms), attack.external_references[0].url))
    return tactics_Outcomes[tacticsnames[0]](node, known_nodes, attack, getParentTechnique(attack))


def getVulns(node):
    """
    Returns actual vulnerabilities from attacks from ATT&CK framework
    """

    attacks = getAttacks(node)
    n = len(attacks)
    ret_vulns = {}
    for i in range(n):
        attack = attacks[i]
        name = attack.name
        print("%d/%d" % (i + 1, n), end=" ")
        outcome = getOutcome(node, attack)
        if outcome is None:  # Set outcome to None to cancel vuln
            print(">> NO VALID OUTCOME")
            continue
        print(">>> FOUND AN OUTCOME")

        ret_vulns[name] = m.VulnerabilityInfo(
            description=attack.description,
            type=getVulnType(attack),
            URL=attack.external_references[0].url,
            outcome=outcome,
            precondition=getPrecondition(attack),
            cost=evaluateVulnCost(attack)
        )
    print("Sucessfully imported %d vulnerabilites into node \"%s\"" % (len(ret_vulns), node['NodeID']))
    return ret_vulns