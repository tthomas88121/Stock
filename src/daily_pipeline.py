from update_price_data import main as update_price_data_main
from model_train import main as train_model_main
from predict import main as predict_main
from evaluate_predictions import evaluate_predictions


def main():
    print("===================================")
    print("STEP 1: Updating stock price data")
    print("===================================")
    update_price_data_main()

    print("\n===================================")
    print("STEP 2: Training models")
    print("===================================")
    train_model_main()

    print("\n===================================")
    print("STEP 3: Generating predictions")
    print("===================================")
    predict_main(top_n=10)

    print("\n===================================")
    print("STEP 4: Evaluating old predictions")
    print("===================================")
    evaluate_predictions()

    print("\n===================================")
    print("Daily pipeline completed.")
    print("===================================")


if __name__ == "__main__":
    main()