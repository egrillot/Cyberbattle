

from ..env.samples.little_network import get_little_environment_profiles, get_machine_list
from ..env.utils.machine import get_machines_by_name
from .attacks import AttackSet

num_client = 5
machines_list = get_machine_list(num_client)
profiles = get_little_environment_profiles(num_client)

def test_affectation_little_environment():

    attacks = AttackSet(machines_list, profiles, max_per_outcomes=1)
    summary_string = attacks.get_attacks_by_machines_string()

    for instance_name, attacks_dict in summary_string.items():

        machine = get_machines_by_name(instance_name, machines_list)[0]
        outcomes = machine.get_outcomes()

        for outcome in outcomes:

            right_required = outcome.get_required_right()
            n = len(attacks_dict.keys())

            if right_required is not None:

                assert sum([1 for attack in attacks_dict.values() if attack['Type'] == 'Local']) == n
            
            else:

                assert sum([1 for attack in attacks_dict.values() if attack['Type'] == 'Remote']) == n

    for i in range(1, 10):

        attacks = AttackSet(machines_list, profiles, max_per_outcomes=i)
        summary_string = attacks.get_attacks_by_machines_string()

        for attacks_dict in summary_string.values():

            assert len(attacks_dict.keys()) <= i


def test_data_sources_for_attacks():

    attacks = AttackSet(machines_list, profiles, max_per_outcomes=1)
    attacks_data_sources = attacks.get_data_sources()
    profiles_data_sources = profiles.get_available_actions()

    assert set(attacks_data_sources).issubset(set(profiles_data_sources))