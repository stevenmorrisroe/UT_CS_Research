from src.nodes import generate_next_idea
from src.config import Config
from src.states import InputState, Assumptions

def test_generate_next_idea():
    try:
        # Setup test config
        config = Config(
            thinking_model="gpt-3.5-turbo",
            run_id="test_run"
        )
        
        # Setup test input state
        input_state = InputState(
            goal="Build a weather-resistant treehouse",
            events=["Purchased lumber", "Drew initial plans"],
            assumptions=Assumptions(
                ground_truth=["Location has mature trees", "Budget is $5000"],
                vulnerabilities=["Weather can be unpredictable"]
            )
        )
        
        print("\nRunning test_generate_next_idea...")
        # Call the function
        result = generate_next_idea(config, input_state)
        
        # Verify the structure of the response
        if not isinstance(result.idea, str):
            raise ValueError(f"Expected idea to be string, got {type(result.idea)}")
        if not isinstance(result.assumptions, list):
            raise ValueError(f"Expected assumptions to be list, got {type(result.assumptions)}")
        if len(result.idea) == 0:
            raise ValueError("Expected non-empty idea")
        if len(result.assumptions) == 0:
            raise ValueError("Expected non-empty assumptions")
        
        # Print results for manual inspection
        print(f"✓ Test passed!")
        print(f"Generated Idea: {result.idea}")
        print(f"Assumptions: {result.assumptions}")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise

def test_generate_next_idea_empty_state():
    try:
        print("\nRunning test_generate_next_idea_empty_state...")
        # Test with minimal input
        config = Config(thinking_model="gpt-3.5-turbo")
        input_state = InputState(
            goal="",
            events=[],
            assumptions=Assumptions(
                ground_truth=[],
                vulnerabilities=[]
            )
        )
        
        result = generate_next_idea(config, input_state)
        if not isinstance(result.idea, str):
            raise ValueError(f"Expected idea to be string, got {type(result.idea)}")
        if not isinstance(result.assumptions, list):
            raise ValueError(f"Expected assumptions to be list, got {type(result.assumptions)}")
            
        print(f"✓ Test passed!")
        print(f"Generated Idea: {result.idea}")
        print(f"Assumptions: {result.assumptions}")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the tests
    try:
        test_generate_next_idea()
        test_generate_next_idea_empty_state()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Tests failed with error: {str(e)}")