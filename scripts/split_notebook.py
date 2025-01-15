import nbformat
import os
import argparse

def split_notebook(input_path, output_dir):
    # Load the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cells = notebook.cells
    segment_cells = []
    segment_count = 1

    for cell in cells:
        segment_cells.append(cell)
        
        # If it's a code cell, save the collected segment
        if cell.cell_type == 'code':
            new_notebook = nbformat.v4.new_notebook()
            new_notebook.cells = segment_cells.copy()
            
            output_file = os.path.join(output_dir, f'segment_{segment_count}.ipynb')
            with open(output_file, 'w', encoding='utf-8') as out_f:
                nbformat.write(new_notebook, out_f)
            
            print(f"Created: {output_file}")
            
            # Reset for the next segment
            segment_cells = []
            segment_count += 1
    
    # Handle any leftover markdown cells
    if segment_cells:
        new_notebook = nbformat.v4.new_notebook()
        new_notebook.cells = segment_cells.copy()
        output_file = os.path.join(output_dir, f'segment_{segment_count}.ipynb')
        with open(output_file, 'w', encoding='utf-8') as out_f:
            nbformat.write(new_notebook, out_f)
        print(f"Created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a Jupyter Notebook into multiple notebooks.")
    parser.add_argument("input_path", help="Path to the input notebook (.ipynb)")
    parser.add_argument("output_dir", help="Directory to save split notebooks")
    
    args = parser.parse_args()
    split_notebook(args.input_path, args.output_dir)
