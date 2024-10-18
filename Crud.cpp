#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Customer {
public:
    int id;
    string name;

    Customer(int id, string name) : id(id), name(name) {}
};

class CustomerService {
private:
    vector<Customer> customers;

public:
    void addCustomer(int id, string name) {
        customers.push_back(Customer(id, name));
        cout << "Customer added: " << name << endl;
    }

    void deleteCustomer(int id) {
        for (auto it = customers.begin(); it != customers.end(); ++it) {
            if (it->id == id) {
                cout << "Customer deleted: " << it->name << endl;
                customers.erase(it);
                return;
            }
        }
        cout << "Customer not found!" << endl;
    }

    void listCustomers() {
        for (const auto& customer : customers) {
            cout << "Customer ID: " << customer.id << ", Name: " << customer.name << endl;
        }
    }
};

int main() {
    CustomerService service;
    service.addCustomer(1, "John Doe");
    service.addCustomer(2, "Jane Smith");

    cout << "All customers:" << endl;
    service.listCustomers();

    service.deleteCustomer(1);

    cout << "All customers after deletion:" << endl;
    service.listCustomers();

    return 0;
}
