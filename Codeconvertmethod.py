import re

def convert_cpp_to_springboot(cpp_code):
    """
    Convert C++ code into a Java Spring Boot microservice structure
    """
    try:
        # Parse the C++ class name (e.g., CustomerService)
        class_pattern = r'class\s+(\w+)\s*\{'
        class_match = re.search(class_pattern, cpp_code)
        class_name = class_match.group(1) if class_match else "UnknownClass"

        # Parse methods (functions)
        method_pattern = r'void\s+(\w+)\s*(.*?)\s*\{'
        methods = re.findall(method_pattern, cpp_code)

        # Example mappings from C++ methods to Java Spring Boot HTTP methods
        method_mappings = {
            'addCustomer': 'POST',
            'deleteCustomer': 'DELETE',
            'listCustomers': 'GET'
        }

        # Spring Boot Controller Template
        controller_template = f"""
        import org.springframework.web.bind.annotation.*;
        import org.springframework.beans.factory.annotation.Autowired;

        @RestController
        @RequestMapping("/api/customers")
        public class {class_name}Controller {{

            @Autowired
            private {class_name}Service {class_name.lower()}Service;

        """

        # Generate method mappings
        for method, params in methods:
            http_method = method_mappings.get(method, 'GET')  # Default to GET
            if method == "addCustomer":
                controller_template += f"""
                @PostMapping
                public String {method}(@RequestBody Customer customer) {{
                    {class_name.lower()}Service.{method}(customer.getId(), customer.getName());
                    return "Customer added";
                }}
                """
            elif method == "deleteCustomer":
                controller_template += f"""
                @DeleteMapping("/{params.split(',')[0].strip()}")
                public String {method}(@PathVariable int id) {{
                    {class_name.lower()}Service.{method}(id);
                    return "Customer deleted";
                }}
                """
            elif method == "listCustomers":
                controller_template += f"""
                @GetMapping
                public List<Customer> {method}() {{
                    return {class_name.lower()}Service.{method}();
                }}
                """

        # Close the controller class
        controller_template += """
        }
        """

        # Service Class Template
        service_template = f"""
        import org.springframework.stereotype.Service;
        import java.util.ArrayList;
        import java.util.List;

        @Service
        public class {class_name}Service {{

            private List<Customer> customers = new ArrayList<>();

            public void addCustomer(int id, String name) {{
                customers.add(new Customer(id, name));
            }}

            public void deleteCustomer(int id) {{
                customers.removeIf(customer -> customer.getId() == id);
            }}

            public List<Customer> listCustomers() {{
                return customers;
            }}
        }}
        """

        # Customer Class Template (DTO)
        customer_template = """
        public class Customer {
            private int id;
            private String name;

            public Customer(int id, String name) {
                this.id = id;
                this.name = name;
            }

            public int getId() {
                return id;
            }

            public String getName() {
                return name;
            }

            public void setId(int id) {
                this.id = id;
            }

            public void setName(String name) {
                this.name = name;
            }
        }
        """

        return controller_template + "\n\n" + service_template + "\n\n" + customer_template

    except Exception as e:
        logger.error(f"An error occurred while converting C++ to Spring Boot: {e}", exc_info=True)
        return None


-----


if file is not None:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing and converting C++ code..."):
                code_content = file.read().decode("utf-8")

                # Update: Call the conversion function
                converted_code = convert_cpp_to_springboot(code_content)

                if converted_code:
                    st.subheader("Generated Java Spring Boot Microservice Code")
                    st.code(converted_code, language='java')
                else:
                    st.error("Failed to convert C++ to Java Spring Boot")

        except Exception as e:
            logger.error(f"An error occurred while processing the code: {e}", exc_info=True)
            st.error(str(e))
else:
    st.info("Please upload a C++ code file to start analyzing.")
