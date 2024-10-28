from wfomc.problems import WFOMCProblem, MLN_to_WFOMC
from .wfomcs_parser import parse as wfomcs_parse
from .mln_parser import parse as mln_parse

def parse_input(input_file: str) -> WFOMCProblem:
    if input_file.endswith('.mln'):
        with open(input_file, 'r') as f:
            input_content = f.read()
        mln_problem = mln_parse(input_content)
        wfomcs_problem = MLN_to_WFOMC(mln_problem)
        return wfomcs_problem
    elif input_file.endswith('.wfomcs'):
        with open(input_file, 'r') as f:
            input_content = f.read()
        return wfomcs_parse(input_content)
    else:
        raise RuntimeError(f'Unknown input file type: {input_file}')
