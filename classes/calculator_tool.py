import logging
from datetime import datetime
from dateutil import parser
from langchain_core.tools import BaseTool

# Define the AgeCalculatorTool class
class AgeCalculatorTool(BaseTool):
    name = "Age Calculator"
    description = """Use this tool when user want to calculate his age. 
    The input to this tool should be a string that will be the birthdate in YYYY-MM-DD only if provided by the user. 
    Never take birthdate as input other than the mentioned format.
    If user does not provide the birthdate do not just assume the birthdate. (Strictly Follow this rule)
    If user does not provide the birthdate pass the NAN value as a string in the input.
    """
    
    def _run(self, birth_date: str):
        try:
            if birth_date=="NAN":
                return "I need your birthdate to calculate the age"
            birth_date = parser.parse(birth_date)
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return str(age)
        except Exception as e:
            #logging.error(f"Error calculating age: {str(e)}")
            return {'error': f"Error calculating age: {str(e)}"}

    def _arun(self, birth_date: str):
        raise NotImplementedError("This tool does not support async operations.")