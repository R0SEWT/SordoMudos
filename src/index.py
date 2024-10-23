from __init__ import test_model

if __name__ == '__main__':
    image = "../images/r_valeria_2.jpg"
    predictions = test_model(image)

    labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 
              9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 
              17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}
    print(f"La letra predicha es: {labels[predictions]}")