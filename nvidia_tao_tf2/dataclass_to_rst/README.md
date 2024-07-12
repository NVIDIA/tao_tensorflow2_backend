# Instructions to generate RST Tables from Dataclasses

## Prerequisites
1. Visual Studio Code (VSCode)
2. VS Code Extension pack for reStructuredText (reStructuredText) : https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext-pack

## Steps
1. Install the extension pack for reStructuredText
2. Open the VSCode terminal and navigate to the directory where the dataclass_to_rst.py file is located
3. Run the following command to generate the RST file from the dataclass file:
    ```
    python dataclass_to_rst.py <module_path> <output_file>
    ```
4. You can use `python dataclass_to_rst.py -h` to get the list of available command line arguments
5. The RST file will be generated in the specified output file path
6. Optional paratemers can be added to the dataclass file by specifying them in the command line arguments
7. Use `--metadata <metadata>` to add metadata to the RST file
8. To format the RST file, press `Tab` for each rst-table to properly format the table