Convert the following C++ code into Java Spring Boot. Generate separate classes only if needed by the logic of the C++ code, avoiding unnecessary layers. Only generate a separate Controller, Service, and Repository if the C++ code includes logic for handling HTTP requests, database interactions, or business logic. If the code is simple (e.g., "Hello World"), convert it within a single MainApplication class.

Ensure to:
1. Conditionally create classes for each layer (e.g., 'Controller', 'Service', 'Entity', 'Repository') based on the complexity of the C++ code.
2. Include 'application.yaml' and 'pom.xml' with only the required dependencies.
3. Annotate each class appropriately (e.g., '@SpringBootApplication' for the main class, '@RestController' for controllers, '@Service' for services, '@Entity' for entities, and '@Repository' for repositories).
4. Ensure that each class is annotated as specified, with no omissions.
5. Avoid generating duplicate code; ensure each logic appears only once.
6. Generate distinct downloadable files for each of the following: Controller, Service, Repository, Entity, the main Spring Boot application class (which contains the public static void main method), pom.xml with needed dependencies, and application.yaml file if applicable.
7. Ensure that there is only one main class in the conversion which has the public static void main method.

Here is the C++ code snippet:
{cpp_code}
