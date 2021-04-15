from ptflops import get_model_complexity_info


def print_info(label: str, model):
    flops, params = get_model_complexity_info(model, (1, 64, 64), False, False)
    print('{} \t {} \t {}'.format(label, flops, params))
