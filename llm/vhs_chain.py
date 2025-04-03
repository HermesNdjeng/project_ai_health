from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import logger
from llm.vhs_schema import VHSInterpretation

def vhs_interpreter_chain() -> Runnable[Any, VHSInterpretation]:
    """Create a chain that provides VHS score interpretation.
    
    Returns:
        The VHS interpretation chain.
    """
    # Set up the LLM
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        response_format={"type": "json_object"}  # This is important!
    )
    
    # Set up the output parser
    vhs_parser = JsonOutputParser(pydantic_object=VHSInterpretation)
    
    # Create the prompt template
    vhs_prompt = """You are a veterinary cardiology expert analyzing radiographic measurements of vertebral heart score (VHS).

## VHS Background:
The Vertebral Heart Score (VHS) is a radiographic measurement that compares heart size to vertebral body length. 
- In dogs, the normal VHS is 9.7 ± 0.5 vertebrae (range: 8.7-10.7)
- In cats, the normal VHS is 7.5 ± 0.3 vertebrae (range: 7.0-8.1)
- The long axis (L) measurement runs from the carina to the cardiac apex
- The short axis (S) measurement is perpendicular to L at the widest part of the heart
- T is the reference vertebral length measurement
- VHS = 6 * ((L + S) / T), expressed in vertebral body units

Some breeds have different normal ranges:
- Boxers, Labrador Retrievers, and Cavalier King Charles Spaniels often have higher normal VHS
- Greyhounds and other deep-chested breeds often have lower normal VHS

## Patient Data:
- Animal Type: {animal_type}
- L Value: {l_value} units
- S Value: {s_value} units
- T Value: {t_value} units
- VHS Score: {vhs_score} vertebrae
- Breed: {breed}
- Age: {age} years
- Weight: {weight} kg
- Sex: {sex}

## Task:
Analyze the VHS measurements and provide a detailed clinical interpretation of the heart size.
Your response MUST be in valid JSON format with the structure specified below.

Your response should have the following format:
{format_instructions}

## VHS ANALYSIS:
"""
    
    vhs_prompt_template = PromptTemplate(
        template=vhs_prompt,
        input_variables=[
            "animal_type",
            "l_value",
            "s_value",
            "t_value",
            "vhs_score",
            "breed",
            "age",
            "weight",
            "sex"
        ],
        partial_variables={"format_instructions": vhs_parser.get_format_instructions()},
    )
    
    # Create and return the chain
    chain: Runnable[Any, VHSInterpretation] = vhs_prompt_template | llm | vhs_parser
    return chain

COMMON_VHS_INTERPRETER_CHAIN = vhs_interpreter_chain()

def interpret_vhs(l_value: float, s_value: float, t_value: float, animal_type: str, 
                 breed: str = None, age: float = None, 
                 weight: float = None, sex: str = None) -> VHSInterpretation:
    """
    Interpret VHS score based on measurements and animal characteristics.
    
    Args:
        l_value: Long axis measurement in units
        s_value: Short axis measurement in units
        t_value: Reference vertebral length in units
        animal_type: "Dog" or "Cat"
        breed: Optional breed information
        age: Optional age in years
        weight: Optional weight in kg
        sex: Optional sex (Male/Female)
        
    Returns:
        Structured VHS interpretation
    """
    # Calculate VHS using the new formula
    vhs_score = 6 * ((l_value + s_value) / t_value)
    logger.info(f"Input measurements - L: {l_value:.2f}, S: {s_value:.2f}, T: {t_value:.2f}, VHS: {vhs_score:.2f}")
    
    # Handle None values properly
    if breed == "":
        breed = None
    
    # Log patient characteristics
    if breed:
        logger.info(f"Breed: {breed}")
    if age:
        logger.info(f"Age: {age:.1f} years")
    if weight:
        logger.info(f"Weight: {weight:.1f} kg")
    if sex:
        logger.info(f"Sex: {sex}")
    
    # Prepare input data
    input_data = {
        "animal_type": animal_type,
        "l_value": l_value,
        "s_value": s_value,
        "t_value": t_value,
        "vhs_score": vhs_score,
        "breed": breed if breed else "Not specified",
        "age": age if age else "Not specified",
        "weight": weight if weight else "Not specified",
        "sex": sex if sex else "Not specified"
    }
    
    # Use the chain to get interpretation
    logger.info("Calling LLM chain for VHS interpretation")
    llm_start_time = datetime.now()
    
    result_dict = COMMON_VHS_INTERPRETER_CHAIN.invoke(input_data)
    
    llm_time = (datetime.now() - llm_start_time).total_seconds()
    logger.info(f"LLM response received in {llm_time:.2f} seconds")
    
    # Log interpretation results
    logger.info(f"Interpretation: {result_dict.get('interpretation', 'N/A')}")
    logger.info(f"Normal range: {result_dict.get('normal_range', 'N/A')}")
    
    if 'severity' in result_dict:
        logger.info(f"Severity: {result_dict['severity']}")
    
    if 'possible_conditions' in result_dict:
        conditions = ', '.join(result_dict['possible_conditions'])
        logger.info(f"Possible conditions: {conditions}")
    
    total_time = (datetime.now() - llm_start_time).total_seconds()
    logger.info(f"VHS interpretation completed in {total_time:.2f} seconds")
    
    result = VHSInterpretation(**result_dict)
    return result


# Example usage
if __name__ == "__main__":
    # Example for a dog with slightly enlarged heart
    dog_interpretation = interpret_vhs(
        l_value=6.2,
        s_value=4.9,
        t_value=1.0,  # Added T value
        animal_type="Dog",
        breed="Labrador Retriever",
        age=8.5,
        weight=32.4,
        sex="Male"
    )
    print("DOG VHS INTERPRETATION:")
    print(dog_interpretation.model_dump_json(indent=2))
    
    # Example for a cat with normal heart
    cat_interpretation = interpret_vhs(
        l_value=4.1,
        s_value=3.3,
        t_value=1.0,  # Added T value
        animal_type="Cat",
        age=5.2,
        weight=4.1,
        sex="Female"
    )
    print("\nCAT VHS INTERPRETATION:")
    print(cat_interpretation.model_dump_json(indent=2))