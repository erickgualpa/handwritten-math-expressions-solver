import numpy as np
import json
import wolframalpha

WOLFRAM_CREDS_FILE = "wolfram-creds.json"
WOLFRAM_OPERATIONS_SYMBOLS = ('*', '/', '[', ']', '(', ')')
INTERNAL_OPERATIONS_SYMBOLS = ('+', '-')

# Convert the symbols list to a unique string
def convert_expression_to_string(math_exp):
    str_math_exp = ''
    for symbol in math_exp:
        # TODO: Añadir un mapeo de símbolos
        str_math_exp += symbol
    return str_math_exp

# Internal Solving
def solve_expression_internally(math_exp):
    return eval(math_exp)

# Wolfram Solving
def connect_to_wolfram_api():
    with open(WOLFRAM_CREDS_FILE, "r") as wolfram_creds_file:
        api_creds = json.load(wolfram_creds_file)

    # print('-- WOLFRAM API CREDS:')
    # print('- app-name: ', api_creds['app-name'])
    # print('- app-id: ', api_creds['app-id'])
    # print('- usage-type: ', api_creds['usage-type'])

    return wolframalpha.Client(api_creds['app-id'])

def solve_expression_on_wolfram_api(str_math_exp):
    client = connect_to_wolfram_api()

    q = "Solve " + str_math_exp
    result = client.query(q)
    answer = next(result.results).text

    return answer

# Decide solving method and solve the expression
def solve_expression(math_exp):
    math_exp = np.array(math_exp)

    # Check if the expression contains complex symbols
    intersect = np.intersect1d(math_exp, WOLFRAM_OPERATIONS_SYMBOLS)

    str_math_exp = convert_expression_to_string(math_exp)
    if len(intersect) > 0:
        result = solve_expression_on_wolfram_api(str_math_exp)
    else:
        result = solve_expression_internally(str_math_exp)

    return str_math_exp, result


