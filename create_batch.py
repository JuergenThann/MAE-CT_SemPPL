
import re
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
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


def _servers(server_device):
    assert isinstance(server_device, str)
    assert re.match(r'^\d+,\d+$', server_device), f"server is invalid, has to consist of a server number and a device number separated by a comma, not '{server_device}'"
    return tuple(int(sd) for sd in server_device.split(','))


@dataclass
class BatchArgs:
    template: str
    output_name: str
    servers: list


def main():
    # parse cli_args immediately for fast cli_args validation
    parser = ArgumentParser()
    parser.add_argument("-t", "--template", type=_yaml, required=True)
    parser.add_argument("-o", "--output_name", type=_foldername, required=True)
    parser.add_argument("-s", "--servers", type=_servers, required=True, nargs='+', dest='servers')
    parsed_args, unparsed_args = parser.parse_known_args()
    batch_args = BatchArgs(**vars(parsed_args))

    assert len(unparsed_args) % 2 == 0
    parameters = []
    for parameter, values in zip(unparsed_args[::2], unparsed_args[1::2]):
        parameter = parameter.replace('--', '')
        values = values.split(',')
        if all(re.match(r'^\d+$', v) for v in values):
            values = [int(v) for v in values]
        elif all(re.match(r'^[\d\.]+$', v) for v in values):
            values = [float(v) for v in values]
        elif all(re.match(r'^(True|False)$', v, re.RegexFlag.IGNORECASE) for v in values):
            values = [v.lower() == 'true' for v in values]
        parameters.append((parameter, values))

    all_parameter_sets = list(product(*(v for _, v in parameters)))
    num_gpus = len(batch_args.servers)

    with open(batch_args.template, 'r') as stream:
        yaml_content = yaml.safe_load(stream)

    final_yamls = []
    for _ in batch_args.servers:
        final_yamls.append(dict(
            stages={},
            ignore_specific_stage_names=True
        ))

    for idx, parameter_values in enumerate(all_parameter_sets):
        modified_yaml = deepcopy(yaml_content)

        current_parameters = zip((p for p, _ in parameters), parameter_values)
        for current_parameter, current_value in current_parameters:
            if isinstance(current_value, bool):
                if not current_value:
                    modified_yaml['vars'][current_parameter] = None
            else:
                modified_yaml['vars'][current_parameter] = current_value

        final_yamls[idx % num_gpus]['stages'][f'execution{idx//num_gpus+1}'] = modified_yaml

    for idx, final_yaml in enumerate(final_yamls):
        server, device = batch_args.servers[idx]
        output_path = Path(batch_args.template).parent / 'batch' / batch_args.output_name / f'{Path(batch_args.template).stem}_student{server:02d}_dev{device:01d}.yaml'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            yaml.dump(final_yaml, f, default_flow_style=False, default_style=None)


if __name__ == "__main__":
    main()
