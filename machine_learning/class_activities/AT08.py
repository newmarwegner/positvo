# Atividade em aula - AT08
# Autor: Newmar Wegner
# Date: 10/04/2021

'''
ATIVIDADE
1. Gerar modelo simple logistic no weka
2. Criar um classificador para diabetes
'''


def predict_negative_diabetes(var):
    equation = (4.2 + var[0] * -0.06 + var[1] * -0.02 + var[2] * 0.01 + var[3] * 0
                + var[4] * 0 + var[5] * -0.04 + var[6] * -0.47 + var[7] * -0.01)
    
    return 1 / (1 + (2.718281 ** -equation))


def predict_positive_diabetes(var):
    equation = (4.2 + var[0] * -0.06 + var[1] * -0.02 + var[2] * 0.01 + var[3] * 0
                + var[4] * 0 + var[5] * -0.04 + var[6] * -0.47 + var[7] * -0.01)
    
    return 1 / (1 + (2.718281 ** equation))


if __name__ == '__main__':
    var_positive = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    var_negative = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    test_negative = predict_negative_diabetes(var_positive)
    test_positive = predict_positive_diabetes(var_positive)
    print(f'Função predict negative retornou {test_negative} e função predict positive retornou {test_positive} com '
          f'uma variavel que na base está acusando como positiva')
