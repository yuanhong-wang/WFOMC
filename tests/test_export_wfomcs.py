import pytest
from pathlib import Path
import tempfile

from wfomc import parse_input
from wfomc.parser.wfomcs_parser import parse as wfomcs_parse


def test_export_simple_wfomcs():
    """Test exporting a simple WFOMC problem"""
    # Parse an existing .wfomcs file
    model_path = Path(__file__).parent.parent / 'models' / 'nonisolated_graph.wfomcs'
    problem = parse_input(str(model_path))
    
    # Export it
    exported = problem.export_wfomcs()
    
    # Check that the export contains expected elements
    assert 'domain' in exported or 'V' in exported
    assert '\\forall' in exported
    
    # Try to parse the exported content back
    re_parsed = wfomcs_parse(exported)
    
    # The sentence should be equivalent
    assert str(problem.sentence) == str(re_parsed.sentence)
    assert problem.domain == re_parsed.domain


def test_export_with_weights():
    """Test exporting a WFOMC problem with weights"""
    model_path = Path(__file__).parent.parent / 'models' / 'friends-smokes.mln'
    if not model_path.exists():
        pytest.skip("friends-smokes.mln not found")
    
    problem = parse_input(str(model_path))
    
    # Export it
    exported = problem.export_wfomcs()
    
    # Check that weights are included
    assert 'aux' in exported or '@aux' in exported
    

def test_export_with_cardinality_constraints():
    """Test exporting a WFOMC problem with cardinality constraints"""
    model_files = list((Path(__file__).parent.parent / 'models').glob('*.wfomcs'))
    
    # Find a file with cardinality constraints
    for model_file in model_files:
        content = model_file.read_text()
        if '|' in content and any(op in content for op in ['<=', '>=', '=', '<', '>']):
            problem = parse_input(str(model_file))
            if problem.cardinality_constraint is not None and not problem.cardinality_constraint.empty():
                exported = problem.export_wfomcs()
                
                # Check that cardinality constraints are included
                assert '|' in exported
                break


def test_export_with_unary_evidence():
    """Test exporting a WFOMC problem with unary evidence"""
    model_files = list((Path(__file__).parent.parent / 'models' / 'linear_order_unary_evidence').glob('*.wfomcs'))
    
    if len(model_files) == 0:
        pytest.skip("No unary evidence files found")
    
    for model_file in model_files:
        problem = parse_input(str(model_file))
        if problem.unary_evidence is not None and len(problem.unary_evidence) > 0:
            exported = problem.export_wfomcs()
            
            # Check that unary evidence is included
            # Evidence should be in the last section
            lines = exported.strip().split('\n')
            # Should contain atomic formulas or negated atomic formulas
            assert any('(' in line and ')' in line for line in lines[-3:])
            break


def test_export_to_file():
    """Test exporting a WFOMC problem to a file"""
    model_path = Path(__file__).parent.parent / 'models' / 'nonisolated_graph.wfomcs'
    problem = parse_input(str(model_path))
    
    # Export to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.wfomcs', delete=False) as f:
        temp_path = f.name
    
    try:
        result = problem.export_wfomcs(temp_path)
        
        # Check that file was created
        assert Path(temp_path).exists()
        
        # Check that the file content matches the returned string
        with open(temp_path, 'r') as f:
            file_content = f.read()
        assert file_content == result
        
        # Check that we can parse the exported file
        re_parsed = parse_input(temp_path)
        assert str(problem.sentence) == str(re_parsed.sentence)
    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_roundtrip():
    """Test that exporting and re-importing preserves the problem"""
    model_path = Path(__file__).parent.parent / 'models' / 'permutation-no-fix-sc2.wfomcs'
    original_problem = parse_input(str(model_path))
    
    # Export and re-import
    exported = original_problem.export_wfomcs()
    re_parsed = wfomcs_parse(exported)
    
    # Check that key properties are preserved
    assert str(original_problem.sentence) == str(re_parsed.sentence)
    assert original_problem.domain == re_parsed.domain
    
    # Check weights (if any)
    if original_problem.weights:
        assert len(original_problem.weights) == len(re_parsed.weights)
