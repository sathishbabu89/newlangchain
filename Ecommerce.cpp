#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>

class Product {
public:
    std::string name;
    double price;

    Product(const std::string &name, double price) : name(name), price(price) {}
};

class Order {
public:
    std::vector<Product> products;
    double total;

    Order() : total(0.0) {}

    void addProduct(const Product &product) {
        products.push_back(product);
        total += product.price;
    }

    void displayOrder() {
        std::cout << "Order Summary:\n";
        for (const auto &product : products) {
            std::cout << "- " << product.name << ": $" << product.price << std::endl;
        }
        std::cout << "Total: $" << total << std::endl;
    }
};

class EcommerceApp {
private:
    std::vector<Product> inventory;

public:
    EcommerceApp() {
        // Adding some products to the inventory
        inventory.emplace_back("Laptop", 999.99);
        inventory.emplace_back("Smartphone", 499.99);
        inventory.emplace_back("Headphones", 199.99);
    }

    void displayProducts() {
        std::cout << "Available Products:\n";
        for (size_t i = 0; i < inventory.size(); ++i) {
            std::cout << i + 1 << ". " << inventory[i].name << " - $" << inventory[i].price << std::endl;
        }
    }

    void processOrder() {
        Order order;
        int choice;

        while (true) {
            displayProducts();
            std::cout << "Enter the product number to add to your order (0 to finish): ";
            std::cin >> choice;

            if (choice == 0) break;

            if (choice > 0 && choice <= inventory.size()) {
                order.addProduct(inventory[choice - 1]);
                std::cout << "Added " << inventory[choice - 1].name << " to your order.\n";
            } else {
                std::cout << "Invalid choice, please try again.\n";
            }
        }

        order.displayOrder();
    }
};

int main() {
    srand(static_cast<unsigned>(time(0))); // Seed for random number generation

    EcommerceApp app;
    app.processOrder();

    return 0;
}
