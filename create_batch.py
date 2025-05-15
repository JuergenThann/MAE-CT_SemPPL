
import re
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from itertools import product, chain
import yaml
from pathlib import Path


def _yaml(yaml_path):
    assert isinstance(yaml_path, str)
    path = Path(yaml_path).expanduser().with_suffix(".yaml")
    assert path.exists(), f"yaml file '{yaml_path}' doesn't exist"
    return yaml_path


def _foldername(name):
    assert isinstance(name, str)
    invalid_chars = re.sub(r'[\w\. _-]+', '', name)
    assert len(invalid_chars) == 0, f"folder name '{name}' contains invalid characters '{invalid_chars}'"
    return name


@dataclass
class BatchArgs:
    template: str
    output_name: str
    simulate: bool


def parse_unparsed_args(unparsed_args):
    debug = False

    parameters = []
    combined_args = []
    current_command_name = None
    current_command_args = None
    for arg in unparsed_args:
        if arg.startswith('--'):
            if current_command_name is not None:
                combined_args.append((current_command_name, current_command_args))
            current_command_name = arg.replace('--', '')
            current_command_args = []
        else:
            if current_command_name != 'replace':
                if debug:
                    print(f'{current_command_name=}')
                    print(f'{arg=}')
                values = arg.split(',')
                if all(re.match(r'^\d+$', v) for v in values):
                    values = [int(v) for v in values]
                    if debug:
                        print('all int')
                elif all(re.match(r'^[\d\.]+$', v) for v in values):
                    values = [float(v) for v in values]
                    if debug:
                        print('all float')
                elif debug:
                    print('mixed types')
                values = [None if v == 'null' else v for v in values]
            else:
                values = arg
            current_command_args.append(values)
    if current_command_name is not None:
        combined_args.append((current_command_name, current_command_args))

    for command, command_args in combined_args:
        if command == 'zip':
            assert len(command_args) % 2 == 0, 'zip arguments have to be specified in pairs of parameter name and values'
            assert all(len(v) == 1 for v in command_args[::2]), 'parameter names cannot be lists'
            assert all(len(v) == len(command_args[1]) for v in command_args[1::2]), 'all value lists have to have the same number of values'
            parameter = tuple(p[0] for p in command_args[::2])
            values = list(zip(*command_args[1::2]))
        elif command == 'replace':
            assert len(command_args) == 2, 'only exactly one parameter with replacement value allowed'
            parameter, replacement = command_args
            assert ':' in replacement, 'value to replace and replacement values have to be separated by colon'
            value_to_replace, replacement_values = replacement.split(':')
            replacement_values = replacement_values.split('/')
            values = [lambda v, ov=value_to_replace, nv=r: re.sub(ov, nv, v) for r in replacement_values]
        elif command == 'remove':
            assert len(command_args) == 2, 'only exactly one parameter with removal boolean allowed'
            parameter, remove = command_args
            parameter = parameter[0]
            remove = [r.lower() for r in remove]
            assert all(r in ('true', 'false') for r in remove), 'only "true" or "false" are allowed for removal boolean'
            remove = [r == 'true' for r in remove]
            values = [lambda v, rem=r: None if rem else v for r in remove]
        else:
            assert len(command_args) == 1, f'only one value list allowed per parameter.'
            parameter = command
            values = command_args[0]
        parameters.append((parameter, values))
    return parameters


def main():
    # parse cli_args immediately for fast cli_args validation
    parser = ArgumentParser()
    parser.add_argument("-t", "--template", type=_yaml, required=True)
    parser.add_argument("-o", "--output_name", type=_foldername, required=True)
    parser.add_argument("-s", "--simulate", action='store_true')
    parsed_args, unparsed_args = parser.parse_known_args()
    batch_args = BatchArgs(**vars(parsed_args))

    parameters = parse_unparsed_args(unparsed_args)

    all_parameter_sets = list(product(*(v for _, v in parameters)))

    execution_prefix = batch_args.template.replace('/', '_').replace('\\', '_').replace('.yaml', '') + '_' + batch_args.output_name
    with open(batch_args.template, 'r') as stream:
        yaml_content = yaml.safe_load(stream)

    batch_yaml = dict(
        stages={},
        ignore_specific_stage_names=True
    )

    parameter_names = list(chain(*(p if isinstance(p, tuple) else (p,) for p, _ in parameters)))
    print()
    for idx, parameter_values in enumerate(all_parameter_sets):
        modified_yaml = deepcopy(yaml_content)

        current_parameter_values = list(chain(*(v if isinstance(v, tuple) else (v,) for v in parameter_values)))
        current_parameters = list(zip(parameter_names, current_parameter_values))
        print(f'Zipped parameters and names to: {current_parameters}')

        for current_parameter, current_value in current_parameters:
            if callable(current_value):
                modified_yaml['vars'][current_parameter] = current_value(modified_yaml['vars'][current_parameter])
            else:
                modified_yaml['vars'][current_parameter] = current_value

        batch_yaml['stages'][f'{execution_prefix}_execution{idx+1:03d}'] = modified_yaml

    print(f'Created {len(all_parameter_sets)} runs.')
    print()
    print('To execute, call these commands on the according student servers:')
    print('cd /system/user/studentwork/thann/mae_ct_semppl/code')
    output_path = Path(batch_args.template).parent / 'batch' / f'{Path(batch_args.template).stem}_{batch_args.output_name}.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not batch_args.simulate:
        with output_path.open('w') as f:
            yaml.dump(batch_yaml, f, default_flow_style=False, default_style=None, sort_keys=False)
    linux_hp_path = output_path.as_posix().replace("\\", "/")
    print(f'python main_train.py --hp {linux_hp_path} --skip_if_exists_in_wandb --devices ...')


if __name__ == "__main__":
    main()
