# LinearAlgebraVisualizer-ME540-ECE515
An interactive web application for visualizing fundamental linear algebra concepts with applications in control theory. This tool provides intuitive visualizations and detailed mathematical analysis to help understand core linear algebra concepts used in control systems.
# Linear Algebra Visualizer - ME540 Control Theory Project

=====================================================================

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning repository)

### Installation

#### Option 1: Using pip

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

#### Option 2: Using conda

# Create and activate conda environment
conda create -n la-visualizer python=3.9
conda activate la-visualizer

# Install packages
conda install -c conda-forge streamlit numpy plotly

# Run the application
streamlit run app.py

Alternative: Anaconda Prompt (Windows)

# Open Anaconda Prompt, navigate to project folder
cd path\to\project\folder
streamlit run app.py

The application will automatically open in your default web browser at http://localhost:8501.

VERY IMPORTANT
***Any method you run it with you will need the CMD or Anaconda console to be at the path of app.py***

=====================================================================
Troubleshooting

Common Issues
--------------------------------
Port already in use:
	streamlit run app.py --server.port 8502
--------------------------------
Package installation fails:
	pip install --upgrade pip
	pip install --no-cache-dir -r requirements.txt
--------------------------------
Python not found (Windows CMD):
	Use Anaconda Prompt instead
Or specify full Python path:
	"C:\Users\YourName\anaconda3\python.exe" -m streamlit run app.py (example path)
--------------------------------
Import errors:
    Ensure all files are in the same directory
    Check Python version (3.8+ required)
    Try creating fresh virtual environment
--------------------------------
3D plots not interactive:
    Ensure Plotly is correctly installed
    Check browser compatibility (Chrome/Firefox recommended)

=====================================================================

Verification Commands

# Check installation
python -c "import streamlit, numpy, plotly; print('All packages installed')"

# Check Streamlit version
streamlit --version

=====================================================================

File Descriptions
------------------
    app.py - Main application

        Streamlit web interface

        User input handling

        Visualization selection

        Session state management

        Welcome page and export/import functionality
------------------
    visualizations.py - Core mathematics

        2D/3D visualization functions

        Matrix analysis algorithms

        Eigenvalue/determinant calculations

        High-dimensional analysis functions

        Inverse matrix computation
------------------
    requirements.txt - Dependencies

        Streamlit (web framework)

        NumPy (linear algebra)

        Plotly (visualizations)

=====================================================================

🎯 How to Use
Basic Navigation

    Select visualization from the sidebar dropdown (starts with Welcome page)

    Enter matrix values using table inputs

    Adjust parameters (number of vectors, examples)

    Interact with plots:

        2D: Hover for values, click legend to toggle

        3D: Drag to rotate, scroll to zoom, right-click to pan

    Expand sections for mathematical explanations

Quick Examples

Use the sidebar's "Quick Examples" section to load predefined matrices:

    Rotation: 30° rotation matrix

    Shear: Shear transformation

    Scaling: Non-uniform scaling

    Reflection: Reflection across axis

Randomization

Click 🎲 buttons to generate:

    Random matrices

    Random vectors

    Random symmetric matrices (in high-dimensional calculator)

Export/Import (High-Dimensional Calculator)

    Export as JSON: Download matrix, vectors, and analysis

    Export as Python: Get Python code for the matrix

    Import JSON: Load previously saved matrices

=====================================================================

Available Visualizations
1. Welcome / Instructions

    Location: First option in sidebar dropdown

    What it shows: Project overview, quick start guide, and feature showcase

    Key features:

        Project description

        Quick start instructions

        Feature overview

2. 2D Vector Transformations

    Location: Second option in sidebar dropdown

    What it shows: How a 2×2 matrix transforms vectors in the plane

    Key features:

        Blue arrows: Original vectors

        Red dashed arrows: Transformed vectors

        Unit circle for reference

        Matrix properties (determinant, rank, condition number)

    Math explanations: Click "Matrix Information" expander below the plot

3. 2D Determinant Visualization

    Location: Third option in sidebar dropdown

    What it shows: Determinant as area scaling factor

    Key features:

        Blue area: Unit square (area = 1)

        Red area: Transformed parallelogram (area = determinant)

        Eigenvectors shown as green arrows

    Math explanations: Click "Mathematical Explanation" expander below the plot

4. 3D Vector Transformations

    Location: Fourth option in sidebar dropdown

    What it shows: 3D vector transformations

    Key features:

        Solid arrows: Original basis vectors

        Dashed arrows: Transformed vectors

        Unit sphere for reference

        Full 3D rotation/zoom/pan controls

    Controls guide: Click "3D View Controls" expander below the plot

5. 3D Determinant Visualization

    Location: Fifth option in sidebar dropdown

    What it shows: Determinant as volume scaling in 3D

    Key features:

        Blue wireframe: Unit cube

        Red wireframe: Transformed parallelepiped

        Eigenvectors as green arrows

        Sarrus' rule calculation shown

    Math explanations: Click "Mathematical Explanation" expander

6. 2D Eigenvectors & Eigenvalues

    Location: Sixth option in sidebar dropdown

    What it shows: Eigenvectors and eigenvalues visualization

    Key features:

        Green arrows: Eigenvectors (invariant directions)

        Purple arrows: Eigenvectors scaled by eigenvalues

        Blue dots: Original sample points

        Red dots: Transformed sample points

    Visual guide: Click "Understanding the Visualization" expander

    Math analysis: Click "Eigen Analysis" expander

7. 3D Eigenvectors & Eigenvalues

    Location: Seventh option in sidebar dropdown

    What it shows: 3D eigenvector visualization

    Key features:

        Blue surface: Unit sphere

        Red surface: Transformed ellipsoid

        Colored arrows: Eigenvectors

        Dashed arrows: Scaled eigenvectors

    Visual guide: Click "Understanding 3D Eigen Analysis" expander

    Math analysis: Click "Eigen Analysis Results" expander

8. High-Dimensional Calculator (4D-8D)

    Location: Eighth option in sidebar dropdown

    What it shows: Matrix analysis for dimensions 4-8

    Key features:

        Matrix input with dimension selector (2-8)

        Vector transformation

        Comprehensive analysis in 5 tabs:

            Matrix Properties: Determinant, trace, rank, norms

            Eigen Analysis: Eigenvalues and eigenvectors

            Matrix Inverse: Inverse computation and verification

            Vector Transformation: Detailed vector analysis

            Interpretation: Mathematical meaning and control theory insights

        Export/Import functionality

    Math explanations: Available in all 5 tabs
