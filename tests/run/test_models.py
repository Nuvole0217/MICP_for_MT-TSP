from tests.run.utils import load_cases, model_testing


def test_complex() -> None:
    model_testing(
        load_cases("complex")
    )
    
def test_simple() -> None:
    model_testing(
        load_cases("simple")
    )

test_simple()