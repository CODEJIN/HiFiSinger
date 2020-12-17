import argparse

def Recursive_Parse(args_Dict):
    parsed_Dict = {}
    for key, value in args_Dict.items():
        if isinstance(value, dict):
            value = Recursive_Parse(value)
        parsed_Dict[key]= value

    args = argparse.Namespace()
    args.__dict__ = parsed_Dict
    return args