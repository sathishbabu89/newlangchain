Convert the following C++ code into Java Spring Boot. Follow these guidelines to ensure accurate and structured conversion:

1. **Class Structure**: Generate separate classes only if necessary based on the complexity of the C++ code. Create a `Controller`, `Service`, and `Repository` only if the code involves handling HTTP requests, business logic, or database interactions. For simple examples (e.g., "Hello World"), convert it into a single `MainApplication` class.

2. **Class Annotations**: Ensure each class is annotated correctly:
   - `@SpringBootApplication` for the main class
   - `@RestController` for the controller class
   - `@Service` for service classes
   - `@Entity` for entity classes
   - `@Repository` for repository classes

3. **Avoid Duplication**: Ensure there is no duplicate code across files. Each logical component should be represented only once.

4. **File Structure**: Generate distinct files for the following components:
   - Controller (if applicable)
   - Service (if applicable)
   - Repository (if applicable)
   - Entity (if applicable)
   - Main Spring Boot application class with the `main` method
   - `pom.xml` with the required dependencies
   - `application.yaml` if applicable

5. **Output Clarity**: The generated code should be clear and organized, making it easy to understand and integrate.

Here is the C++ code snippet:
{cpp_code}
