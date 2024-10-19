#include <iostream>
#include <vector>
#include <string>

class User {
public:
    int id;
    std::string name;
    std::string email;

    User(int uid, const std::string &uname, const std::string &uemail)
        : id(uid), name(uname), email(uemail) {}
};

class UserManagement {
private:
    std::vector<User> users;
    int nextId;

public:
    UserManagement() : nextId(1) {}

    void createUser(const std::string &name, const std::string &email) {
        User newUser(nextId++, name, email);
        users.push_back(newUser);
        std::cout << "User created: " << name << std::endl;
    }

    void readUsers() {
        std::cout << "User List:\n";
        for (const auto &user : users) {
            std::cout << "ID: " << user.id << ", Name: " << user.name
                      << ", Email: " << user.email << std::endl;
        }
    }

    void updateUser(int id, const std::string &name, const std::string &email) {
        for (auto &user : users) {
            if (user.id == id) {
                user.name = name;
                user.email = email;
                std::cout << "User updated: " << name << std::endl;
                return;
            }
        }
        std::cout << "User not found!" << std::endl;
    }

    void deleteUser(int id) {
        for (auto it = users.begin(); it != users.end(); ++it) {
            if (it->id == id) {
                std::cout << "User deleted: " << it->name << std::endl;
                users.erase(it);
                return;
            }
        }
        std::cout << "User not found!" << std::endl;
    }
};

int main() {
    UserManagement userManager;

    userManager.createUser("Alice", "alice@example.com");
    userManager.createUser("Bob", "bob@example.com");
    userManager.readUsers();

    userManager.updateUser(1, "Alice Smith", "alice.smith@example.com");
    userManager.readUsers();

    userManager.deleteUser(2);
    userManager.readUsers();

    return 0;
}
