import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.state import SimulationState
from core.simulation import check_seller_action, initialize_simulation
from utils.persona import create_persona_placeholder, get_available_persona_ids

@pytest.fixture
def setup_test_environment():
    """Create test environment with persona data."""
    create_persona_placeholder()
    # Check if persona data was created successfully
    assert Path("data/personas/persona_topic_0_prompt_nmf.txt").exists()
    return get_available_persona_ids()

def test_get_available_persona_ids(setup_test_environment):
    """Test that we can get available persona IDs."""
    personas = get_available_persona_ids()
    assert isinstance(personas, list)
    assert len(personas) > 0
    assert "topic_0" in personas

def test_check_seller_action_empty_messages():
    """Test that check_seller_action handles empty messages correctly."""
    state = SimulationState(messages=[])
    result = check_seller_action(state)
    assert result == "__end__"

def test_initialize_simulation():
    """Test simulation initialization with a mock config."""
    # Create test environment
    create_persona_placeholder()
    
    # Mock state and config
    state = SimulationState(messages=[])
    config = {"configurable": {"persona_id": "topic_0"}}
    
    # Run initialization
    result = initialize_simulation(state, config)
    
    # Check results
    assert "persona_id" in result
    assert result["persona_id"] == "topic_0"
    assert "current_buyer_prompt" in result
    assert "initial_seller_prompt" in result
    assert isinstance(result["current_buyer_prompt"], str)
    assert len(result["current_buyer_prompt"]) > 0 