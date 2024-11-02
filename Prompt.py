Convert the following C++ code into a Java Spring Boot application. Please follow these guidelines carefully to ensure accurate and well-structured conversion:

1. **Class Structure**: 
   - Analyze the C++ code to determine the necessary components. 
   - Create a `Controller`, `Service`, and `Repository` only if the code involves:
     - Handling HTTP requests (create a `Controller`)
     - Implementing business logic (create a `Service`)
     - Interacting with a database (create a `Repository`)
   - For simple C++ programs (e.g., "Hello World"), produce a single `MainApplication` class.

2. **Class Annotations**: Ensure each class is correctly annotated:
   - Use `@SpringBootApplication` for the main application class.
   - Use `@RestController` for the controller class.
   - Use `@Service` for service classes.
   - Use `@Entity` for entity classes.
   - Use `@Repository` for repository classes.

3. **Avoid Duplication**: 
   - Check that there is no duplicate code across files. Each logical component must be represented only once in the generated Java code.

4. **File Structure**: 
   - Generate distinct files for:
     - `Controller` (if applicable)
     - `Service` (if applicable)
     - `Repository` (if applicable)
     - `Entity` (if applicable)
     - The main Spring Boot application class with the `main` method
     - `pom.xml` with all required dependencies
     - `application.yaml` if applicable

5. **Output Clarity**: 
   - The generated code should be clear, organized, and properly formatted, making it easy to read and integrate into a Spring Boot application.

6. **Error Handling**: 
   - If any part of the generated code does not meet the above criteria, please indicate what is missing or incorrect, and ensure all parts are logically connected.

Here is the C++ code snippet:
{cpp_code}
