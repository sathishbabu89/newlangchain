def convert_cpp_to_java_spring_boot(cpp_code, filename, HUGGINGFACE_API_TOKEN, project_info):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(cpp_code)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                max_new_tokens=2048,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
            )

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 4: Converting C++ to Java Spring Boot... 🔄")

            prompt = f"""
Convert the following C++ code into Java Spring Boot. Generate separate classes only if needed by the logic of the C++ code, avoiding unnecessary layers.
Only generate a separate `Controller`, `Service`, and `Repository` if the C++ code includes logic for handling HTTP requests, database interactions, or business logic. If the code is simple (e.g., "Hello World"), convert it within a single `MainApplication` class.
Here is the C++ code snippet:
{cpp_code}
"""
            response = llm.invoke(prompt)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.success("Step 5: Conversion complete! 🎉")

            # Parsing response to identify components
            components = {}
            lines = response.splitlines()
            current_class = None

            for line in lines:
                if line.startswith("public class ") or line.startswith("class "):
                    current_class = line.split()[2].strip()  # Extract class name
                    components[current_class] = []
                if current_class:
                    components[current_class].append(line)

            # Step 6: Generate Spring Boot project
            st.info("Step 6: Generating Spring Boot project... 📦")
            spring_boot_project = generate_spring_boot_project(project_info)
            
            if spring_boot_project:
                # Create a zip buffer for the final download
                zip_buffer = io.BytesIO()
                zip_filename = filename.rsplit('.', 1)[0] + '.zip'
                
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    # Save each converted Java class to the zip file with the correct package structure
                    package_path = project_info['packageName'].replace('.', '/')

                    for class_name, class_lines in components.items():
                        class_code = "\n".join(class_lines)
                        # Determine the class type based on naming conventions
                        if "Controller" in class_name:
                            class_path = f"src/main/java/{package_path}/controller/{class_name}.java"
                        elif "Service" in class_name:
                            class_path = f"src/main/java/{package_path}/service/{class_name}.java"
                        elif "Repository" in class_name:
                            class_path = f"src/main/java/{package_path}/repository/{class_name}.java"
                        else:
                            class_path = f"src/main/java/{package_path}/{class_name}.java"

                        zip_file.writestr(class_path, class_code)

                    # Add Spring Boot project zip content
                    zip_file.writestr("spring-boot-project.zip", spring_boot_project)

                zip_buffer.seek(0)  # Move to the beginning of the BytesIO buffer

                # Download button for the complete project zip file
                st.download_button(
                    label="Download Complete Spring Boot Project",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )

                # Provide individual downloads for each class
                for class_name, class_lines in components.items():
                    class_code = "\n".join(class_lines)
                    st.download_button(
                        label=f"Download {class_name}.java",
                        data=class_code,
                        file_name=f"{class_name}.java",
                        mime="text/x-java-source"
                    )
            else:
                st.error("Failed to generate the Spring Boot project.")

            # Display the converted Java code
            st.code(response, language='java')

            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning("The converted Java code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success("The Java code is free from basic syntax errors!")

    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert C++ code to Java.")
