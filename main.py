# main.py
import os
import sys

def generate_data():
    from src import generate_data
    generate_data.__name__  # triggers the script
    print("Data generation complete!")

def train_model():
    from src import train
    train.__name__  # triggers the script
    print("Training complete!")

def run_inference():
    from src.infer import run_inference as do_inference
    do_inference()

def main():
    print("=== OCR Project Menu ===")
    print("1) Generate synthetic dataset")
    print("2) Train OCR model")
    print("3) Run inference on an image")
    print("4) Exit")

    choice = input("Enter choice (1-4): ").strip()
    if choice == "1":
        generate_data()
    elif choice == "2":
        train_model()
    elif choice == "3":
        run_inference()
    elif choice == "4":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")
    print("\n")
    main()  # loop back to menu

if __name__ == "__main__":
    main()
