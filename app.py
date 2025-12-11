"""
Linear Algebra Visualizer - Streamlit Web App
Main application file for interactive linear algebra visualizations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from visualizations import (
    visualize_2d_transformation,
    visualize_2d_determinant,
    visualize_3d_transformation,
    visualize_3d_determinant,
    visualize_eigen_2d,
    visualize_3d_eigen,
    analyze_high_dim_matrix,
    transform_vectors,
    compute_matrix_inverse,
    compute_matrix_properties
)
import json
import datetime

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Linear Algebra Visualizer",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ INITIALIZE SESSION STATE ============
# Matrix and vector storage
if 'matrix_2d_trans' not in st.session_state:
    st.session_state.matrix_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
    st.session_state.vectors_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
    st.session_state.num_vectors_2d = 2

if 'matrix_2d_det' not in st.session_state:
    st.session_state.matrix_2d_det = [[1.0, 0.0], [0.0, 1.0]]

if 'matrix_3d_trans' not in st.session_state:
    st.session_state.matrix_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    st.session_state.vectors_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    st.session_state.num_vectors_3d = 3

if 'matrix_2d_eig' not in st.session_state:
    st.session_state.matrix_2d_eig = [[2.0, 1.0], [1.0, 2.0]]
    
if 'matrix_3d_det' not in st.session_state:
    st.session_state.matrix_3d_det = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

if 'matrix_3d_eigen' not in st.session_state:
    st.session_state.matrix_3d_eigen = [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.5]]

# High-dimensional calculator session states
if 'high_dim_matrix' not in st.session_state:
    st.session_state.high_dim_matrix = [[1.0, 0.0], [0.0, 1.0]]
if 'high_dim_vectors' not in st.session_state:
    st.session_state.high_dim_vectors = [[1.0, 0.0], [0.0, 1.0]]
if 'high_dim_n' not in st.session_state:
    st.session_state.high_dim_n = 2

# Randomization tracking
if 'randomize_trigger' not in st.session_state:
    st.session_state.randomize_trigger = 0

# ============ RANDOMIZATION FUNCTIONS ============
def get_random_matrix(rows, cols, scale=2):
    """Generate a random matrix with values between -scale and scale"""
    return (np.random.rand(rows, cols) * 2 * scale - scale).tolist()

def get_random_vectors(num_vectors, dim, scale=2):
    """Generate random vectors"""
    return [(np.random.rand(dim) * 2 * scale - scale).tolist() 
            for _ in range(num_vectors)]

# ============ MATRIX INPUT WIDGETS ============
def create_matrix_table(rows, cols, values, key_suffix=""):
    """Create a table-style matrix input with improved formatting"""
    matrix = []
    for i in range(rows):
        cols_widgets = st.columns(cols)
        row_vals = []
        for j in range(cols):
            with cols_widgets[j]:
                # Use a unique key that includes the visualization type
                key = f"mat_{i}_{j}_{key_suffix}"
                val = st.number_input(
                    f"",
                    value=float(values[i][j]),
                    step=0.1,
                    format="%.4f",
                    key=key,
                    label_visibility="collapsed",
                    on_change=lambda i=i, j=j, key_suffix=key_suffix: update_matrix_value(i, j, key_suffix)
                )
                row_vals.append(val)
        matrix.append(row_vals)
    return matrix

def update_matrix_value(i, j, section):
    """Update matrix value in session state when user changes input"""
    key = f"mat_{i}_{j}_{section}"
    if key in st.session_state:
        if section == "2d_trans":
            st.session_state.matrix_2d_trans[i][j] = st.session_state[key]
        elif section == "2d_det":
            st.session_state.matrix_2d_det[i][j] = st.session_state[key]
        elif section == "3d_trans":
            st.session_state.matrix_3d_trans[i][j] = st.session_state[key]
        elif section == "2d_eig":
            st.session_state.matrix_2d_eig[i][j] = st.session_state[key]

def create_vector_input(length, values, key_prefix, index):
    """Create an interactive vector input widget"""
    cols = st.columns(length)
    vector = []
    
    for i in range(length):
        with cols[i]:
            key = f"{key_prefix}_{index}_{i}"
            val = st.number_input(
                f"[{i}]",
                value=float(values[i]),
                step=0.1,
                format="%.4f",
                key=key,
                on_change=lambda i=i, key_prefix=key_prefix, index=index: update_vector_value(i, key_prefix, index)
            )
            vector.append(val)
    
    return vector

def update_vector_value(i, key_prefix, vec_index):
    """Update vector value in session state when user changes input"""
    key = f"{key_prefix}_{vec_index}_{i}"
    if key in st.session_state:
        if key_prefix == "vec_2d":
            if vec_index < len(st.session_state.vectors_2d_trans):
                st.session_state.vectors_2d_trans[vec_index][i] = st.session_state[key]
        elif key_prefix == "vec_3d":
            if vec_index < len(st.session_state.vectors_3d_trans):
                st.session_state.vectors_3d_trans[vec_index][i] = st.session_state[key]

# ============ SIDEBAR ============
with st.sidebar:
    st.title("🧮 Linear Algebra Visualizer")
    st.markdown("---")
    
    visualization_type = st.selectbox(
        "Choose Visualization",
        [
            "Welcome / Instructions",
            "2D Vector Transformations",
            "3D Vector Transformations",
            "2D Determinant Visualization",
            "3D Determinant Visualization",
            "2D Eigenvectors & Eigenvalues",
            "3D Eigenvectors & Eigenvalues",
            "High-Dimensional Calculator (4D-8D)"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Quick Examples")
    
    # Quick examples
    example_type = st.radio(
        "Load Example:",
        ["None", "Rotation", "Shear", "Scaling", "Reflection"],
        horizontal=False
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This interactive tool visualizes linear algebra concepts:
    - Vector transformations
    - Determinants as area/volume scaling
    - Eigenvectors & eigenvalues
    - High-dimensional analysis (up to 8D)
    
    Built for ME540 - Control Theory
    """)
    
    # Credits section
    st.markdown("---")
    st.markdown("### Credits")
    st.markdown('<p style="font-size: 12px; color: #666;">Created as a project for ME540<br>Developed with Streamlit, Plotly, and NumPy<br>Made by Osama Rawshdeh - osama4@illinois.edu</p>', 
                unsafe_allow_html=True)

# ============ MAIN CONTENT ============
# ============ WELCOME PAGE ============
if visualization_type == "Welcome / Instructions":
    st.title("🧮 Linear Algebra Visualizer")
    st.markdown("### Interactive Visualizations for Control Theory (ME540)")
    
    # Overview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Explore linear algebra concepts through interactive visualizations:**
        
        - **2D/3D Transformations**: See how matrices transform vectors
        - **Determinants**: Visualize area/volume scaling
        - **Eigen Analysis**: Understand eigenvectors and eigenvalues
        - **High-Dimensional Calculator**: Analyze matrices up to 8×8
        
        **How this connects to Control Theory:**
        
        Linear algebra is the mathematical foundation of control systems:
        - State-space representations use matrices (ẋ = Ax + Bu)
        - Stability analysis relies on eigenvalues
        - Coordinate transformations use linear algebra
        - System properties (controllability, observability) are matrix properties
        """)
    
    with col2:
        # You can add an image or keep it simple
        st.info("**Course**: ME540 - Control Theory")
        st.info("**Project**: Linear Algebra Visualization Tool")
        st.info("**Student**: Osama Rawshdeh")
    
    # Quick start guide
    with st.expander("📋 Quick Start Guide", expanded=True):
        st.markdown("""
        **For the TA evaluating this project:**
        
        1. **Run the app**: `streamlit run app.py`
        2. **Test visualizations**:
           - Review README.txt for setup instructions
           - Select any visualization from the sidebar
           - Click "Randomize Matrix" to see different transformations
           - Try the "Rotation" example from the sidebar
        3. **Explore all features**:
           - 2D/3D Determinant visualizations
           - Eigenvector visualizations
           - High-dimensional calculator (try dimension 4+)
           - Export/Import functionality
        """)
    
    # Feature showcase
    st.markdown("---")
    st.subheader("📊 Available Visualizations")
    
    features = [
        ("2D Transformations", "See how 2×2 matrices transform vectors in the plane", "📐"),
        ("2D Determinants", "Visualize determinants as area scaling factors", "📏"),
        ("3D Transformations", "Interactive 3D visualization of vector transformations", "🧊"),
        ("3D Determinants", "See 3D volume scaling with wireframe cubes", "📦"),
        ("2D Eigen Analysis", "Visualize eigenvectors and eigenvalues in 2D", "🎯"),
        ("3D Eigen Analysis", "3D eigenvector visualization with unit spheres", "🌐"),
        ("High-Dim Calculator", "Analyze 4D-8D matrices with comprehensive results", "🧮"),
    ]
    
    cols = st.columns(3)
    for i, (title, desc, icon) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
    
    # Stop here - don't render other visualizations
    st.stop()

# Only show this title if not on Welcome page
st.title("Interactive Linear Algebra Visualizations")
st.markdown("Visualize core linear algebra concepts with interactive examples")

# ============ 2D VECTOR TRANSFORMATIONS ============
if visualization_type == "2D Vector Transformations":
    st.header("2D Vector Transformations")
    st.markdown("""
    Visualize how a 2×2 matrix transforms vectors in the plane.
    Blue arrows show original vectors, red dashed arrows show transformed vectors.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Handle examples
        if example_type == "Rotation":
            st.session_state.matrix_2d_trans = [[0.866, -0.5], [0.5, 0.866]]
            st.session_state.vectors_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
        elif example_type == "Shear":
            st.session_state.matrix_2d_trans = [[1.0, 0.5], [0.0, 1.0]]
            st.session_state.vectors_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
        elif example_type == "Scaling":
            st.session_state.matrix_2d_trans = [[2.0, 0.0], [0.0, 0.5]]
            st.session_state.vectors_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
        elif example_type == "Reflection":
            st.session_state.matrix_2d_trans = [[-1.0, 0.0], [0.0, 1.0]]
            st.session_state.vectors_2d_trans = [[1.0, 0.0], [0.0, 1.0]]
        
        # Matrix input
        st.markdown("**Transformation Matrix A (2×2)**")
        A = create_matrix_table(2, 2, st.session_state.matrix_2d_trans, key_suffix="2d_trans")
        
        # Vector input - default to 2 basis vectors
        st.markdown("**Vectors to Transform**")
        num_vectors = st.slider("Number of vectors", 1, 5, st.session_state.num_vectors_2d, 
                               key="num_vec_2d")
        st.session_state.num_vectors_2d = num_vectors
        
        # Ensure vectors list has correct length
        if len(st.session_state.vectors_2d_trans) < num_vectors:
            for _ in range(num_vectors - len(st.session_state.vectors_2d_trans)):
                st.session_state.vectors_2d_trans.append([1.0, 1.0])
        elif len(st.session_state.vectors_2d_trans) > num_vectors:
            st.session_state.vectors_2d_trans = st.session_state.vectors_2d_trans[:num_vectors]
        
        vectors = []
        for i in range(num_vectors):
            vector = create_vector_input(2, st.session_state.vectors_2d_trans[i], "vec_2d", i)
            vectors.append(vector)
        
        # Randomization buttons
        st.markdown("**Randomization**")
        rand_col1, rand_col2, rand_col3 = st.columns(3)
        with rand_col1:
            if st.button("🎲 Random Matrix", key="rand_mat_2d"):
                st.session_state.matrix_2d_trans = get_random_matrix(2, 2)
                st.session_state.randomize_trigger += 1
                st.rerun()
        
        with rand_col2:
            if st.button("🎲 Random Vectors", key="rand_vec_2d"):
                st.session_state.vectors_2d_trans = get_random_vectors(num_vectors, 2)
                st.session_state.randomize_trigger += 1
                st.rerun()
        
        with rand_col3:
            if st.button("🎲 Random Both", key="rand_both_2d"):
                st.session_state.matrix_2d_trans = get_random_matrix(2, 2)
                st.session_state.vectors_2d_trans = get_random_vectors(num_vectors, 2)
                st.session_state.randomize_trigger += 1
                st.rerun()
    
    with col2:
        st.subheader("Visualization")
        
        try:
            # Create visualization
            fig = visualize_2d_transformation(A, vectors)
            
            # Display plot with adjusted size
            st.plotly_chart(fig, use_container_width=True, height=600)
            
            # Matrix info in expander
            with st.expander("Matrix Information"):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    det = np.linalg.det(A)
                    st.metric("Determinant", f"{det:.4f}")
                
                with col_info2:
                    rank = np.linalg.matrix_rank(A)
                    st.metric("Rank", rank)
                
                with col_info3:
                    cond = np.linalg.cond(A)
                    st.metric("Condition Number", f"{cond:.4f}")
                
                # Show matrix
                st.markdown("**Matrix:**")
                st.latex(f"""
                A = \\begin{{bmatrix}}
                {A[0][0]:.4f} & {A[0][1]:.4f} \\\\
                {A[1][0]:.4f} & {A[1][1]:.4f}
                \\end{{bmatrix}}
                """)
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Please check your input values.")

# ============ 2D DETERMINANT VISUALIZATION ============
elif visualization_type == "2D Determinant Visualization":
    st.header("2D Determinant Visualization")
    st.markdown("""
    The determinant represents the scaling factor of area.
    Blue region: unit square (area = 1)
    Red region: transformed square (area = determinant)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Matrix")
        st.markdown("Enter a 2×2 matrix:")
        
        # Handle examples
        if example_type == "Rotation":
            st.session_state.matrix_2d_det = [[0.866, -0.5], [0.5, 0.866]]
        elif example_type == "Shear":
            st.session_state.matrix_2d_det = [[1.0, 0.5], [0.0, 1.0]]
        elif example_type == "Scaling":
            st.session_state.matrix_2d_det = [[2.0, 0.0], [0.0, 0.5]]
        elif example_type == "Reflection":
            st.session_state.matrix_2d_det = [[-1.0, 0.0], [0.0, 1.0]]
        
        A = create_matrix_table(2, 2, st.session_state.matrix_2d_det, key_suffix="2d_det")
        
        # Randomization button
        if st.button("🎲 Randomize Matrix", key="rand_det"):
            st.session_state.matrix_2d_det = get_random_matrix(2, 2)
            st.session_state.randomize_trigger += 1
            st.rerun()
    
    with col2:
        st.subheader("Determinant Visualization")
        
        try:
            fig, det_value = visualize_2d_determinant(A)
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True, height=600)
            
            # Mathematical explanation in expander
            with st.expander("Mathematical Explanation"):
                st.markdown(f"""
                The determinant of a 2×2 matrix [[a, b], [c, d]] is calculated as:
                
                $$
                \\det(A) = ad - bc = {A[0][0]:.4f}×{A[1][1]:.4f} - {A[0][1]:.4f}×{A[1][0]:.4f} = {det_value:.4f}
                $$
                
                **Interpretation:**
                - **Area scaling factor**: The transformed area is **{abs(det_value):.4f}×** the original area
                - **Orientation**: {"Positive (preserved)" if det_value >= 0 else "Negative (reversed)"}
                - **Invertibility**: {"Matrix is invertible" if abs(det_value) > 1e-10 else "Matrix is singular (non-invertible)"}
                
                **Geometric meaning:**
                The determinant tells you how much the transformation scales area. 
                A determinant of 0 means the transformation collapses the plane into a lower dimension.
                """)
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

# ============ 3D VECTOR TRANSFORMATIONS ============
elif visualization_type == "3D Vector Transformations":
    st.header("3D Vector Transformations")
    st.markdown("""
    Visualize how a 3×3 matrix transforms vectors in 3D space.
    Solid arrows: original basis vectors
    Dashed arrows: transformed vectors
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        st.markdown("**Transformation Matrix A (3×3)**")
        
        # Handle examples
        if example_type == "Rotation":
            st.session_state.matrix_3d_trans = [[0.866, -0.5, 0.0], [0.5, 0.866, 0.0], [0.0, 0.0, 1.0]]
            st.session_state.vectors_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif example_type == "Scaling":
            st.session_state.matrix_3d_trans = [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.5]]
            st.session_state.vectors_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif example_type == "Shear":
            st.session_state.matrix_3d_trans = [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]
            st.session_state.vectors_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif example_type == "Reflection":
            st.session_state.matrix_3d_trans = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            st.session_state.vectors_3d_trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        A = create_matrix_table(3, 3, st.session_state.matrix_3d_trans, key_suffix="3d_trans")
        
        # Vector input - default to 3 basis vectors
        st.markdown("**Vectors to Transform**")
        num_vectors = st.slider("Number of vectors", 1, 6, st.session_state.num_vectors_3d,
                               key="num_vec_3d")
        st.session_state.num_vectors_3d = num_vectors
        
        # Ensure vectors list has correct length
        if len(st.session_state.vectors_3d_trans) < num_vectors:
            for _ in range(num_vectors - len(st.session_state.vectors_3d_trans)):
                st.session_state.vectors_3d_trans.append([1.0, 1.0, 1.0])
        elif len(st.session_state.vectors_3d_trans) > num_vectors:
            st.session_state.vectors_3d_trans = st.session_state.vectors_3d_trans[:num_vectors]
        
        vectors = []
        for i in range(num_vectors):
            vector = create_vector_input(3, st.session_state.vectors_3d_trans[i], "vec_3d", i)
            vectors.append(vector)
        
        # Randomization buttons
        st.markdown("**Randomization**")
        rand_col1, rand_col2, rand_col3 = st.columns(3)
        with rand_col1:
            if st.button("🎲 Random Matrix", key="rand_mat_3d"):
                st.session_state.matrix_3d_trans = get_random_matrix(3, 3)
                st.session_state.randomize_trigger += 1
                st.rerun()
        
        with rand_col2:
            if st.button("🎲 Random Vectors", key="rand_vec_3d"):
                st.session_state.vectors_3d_trans = get_random_vectors(num_vectors, 3)
                st.session_state.randomize_trigger += 1
                st.rerun()
        
        with rand_col3:
            if st.button("🎲 Random Both", key="rand_both_3d"):
                st.session_state.matrix_3d_trans = get_random_matrix(3, 3)
                st.session_state.vectors_3d_trans = get_random_vectors(num_vectors, 3)
                st.session_state.randomize_trigger += 1
                st.rerun()
    
    with col2:
        st.subheader("3D Visualization")
        
        try:
            fig = visualize_3d_transformation(A, vectors)
            
            # Display 3D plot with enhanced controls
            st.plotly_chart(
                fig, 
                use_container_width=True,
                height=700,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'displaylogo': False
                }
            )
            
            # 3D plot controls info
            with st.expander("3D View Controls"):
                st.markdown("""
                **Mouse Controls:**
                - **Rotate**: Click and drag
                - **Zoom**: Scroll wheel or right-click + drag
                - **Pan**: Right-click + drag or Ctrl + click + drag
                
                **Toolbar Controls:**
                - Home: Reset view
                - Zoom in/out
                - Pan
                - 3D rotation toggle
                """)
            
            # Matrix analysis in expander
            with st.expander("Matrix Analysis"):
                try:
                    det_3d = np.linalg.det(A)
                    eigvals_3d = np.linalg.eigvals(A)
                    
                    col1_info, col2_info, col_info3 = st.columns(3)
                    
                    with col1_info:
                        st.metric("3D Determinant", f"{det_3d:.4f}")
                        st.caption("Volume scaling factor")
                    
                    with col2_info:
                        st.metric("Rank", np.linalg.matrix_rank(A))
                        st.caption("Dimension of column space")
                    
                    with col_info3:
                        eigenvalues_str = ", ".join([f"{v:.4f}" for v in eigvals_3d])
                        st.metric("Eigenvalues", eigenvalues_str)
                    
                    # Show matrix
                    st.markdown("**Matrix:**")
                    st.latex(f"""
                    A = \\begin{{bmatrix}}
                    {A[0][0]:.4f} & {A[0][1]:.4f} & {A[0][2]:.4f} \\\\
                    {A[1][0]:.4f} & {A[1][1]:.4f} & {A[1][2]:.4f} \\\\
                    {A[2][0]:.4f} & {A[2][1]:.4f} & {A[2][2]:.4f}
                    \\end{{bmatrix}}
                    """)
                    
                except:
                    st.warning("Could not compute matrix properties")
            
        except Exception as e:
            st.error(f"Error creating 3D visualization: {str(e)}")

# ============ 2D EIGENVECTORS & EIGENVALUES ============
elif visualization_type == "2D Eigenvectors & Eigenvalues":
    st.header("2D Eigenvectors & Eigenvalues")
    st.markdown("""
    Eigenvectors are special vectors that only get scaled when transformed.
    
    **Visual elements explained:**
    - **Green arrows**: Eigenvectors (direction remains unchanged)
    - **Purple arrows**: Eigenvectors scaled by their eigenvalues (λ × eigenvector)
    - **Blue dots**: Sample points in original space
    - **Red dots**: Transformed sample points
    - **Gray lines**: Connection between original and transformed points
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Matrix")
        st.markdown("Enter a 2×2 matrix (real eigenvalues work best for visualization):")
        
        # Handle examples
        if example_type == "Scaling":
            st.session_state.matrix_2d_eig = [[2.0, 0.0], [0.0, 1.5]]
        elif example_type == "Shear":
            st.session_state.matrix_2d_eig = [[1.0, 0.5], [0.5, 1.0]]
        elif example_type == "Rotation":
            st.session_state.matrix_2d_eig = [[0.866, -0.5], [0.5, 0.866]]
        elif example_type == "Reflection":
            st.session_state.matrix_2d_eig = [[-1.0, 0.0], [0.0, 1.0]]
        
        A = create_matrix_table(2, 2, st.session_state.matrix_2d_eig, key_suffix="2d_eig")
        
        if st.button("🎲 Randomize Matrix", key="rand_eig"):
            st.session_state.matrix_2d_eig = get_random_matrix(2, 2)
            st.session_state.randomize_trigger += 1
            st.rerun()
    
    with col2:
        st.subheader("Eigenvector Visualization")
        
        try:
            fig = visualize_eigen_2d(A)
            
            # Display plot with adjusted size
            st.plotly_chart(fig, use_container_width=True, height=650)
            
            # Trace explanation in expander
            with st.expander("Understanding the Visualization"):
                st.markdown("""
                **What you're seeing:**
                
                **1. Sample Points (Blue & Red)**
                - **Blue dots**: Random points in the original space
                - **Red dots**: Same points after transformation by matrix A
                - **Gray lines**: Connect each point to its transformed version
                
                **2. Eigenvectors (Green arrows)**
                - These are special directions that don't change under transformation
                - They only get scaled by their corresponding eigenvalues
                - Length shows the eigenvector direction (not magnitude)
                
                **3. Scaled Eigenvectors (Purple arrows)**
                - Shows what happens when you multiply eigenvectors by eigenvalues
                - For a true eigenvector: Purple arrow = Green arrow × eigenvalue
                - If the green and purple arrows point in exactly the same/opposite direction, 
                  it's a perfect eigenvector
                
                **Key Insight:**
                Points along eigenvector directions simply move along the same line,
                either stretching or compressing based on the eigenvalue magnitude.
                """)
            
            # Eigen analysis in expander
            with st.expander("Eigen Analysis"):
                try:
                    eigvals, eigvecs = np.linalg.eig(A)
                    
                    # Display eigenvalues and eigenvectors
                    st.markdown("**Eigenvalues:**")
                    col_eig1, col_eig2 = st.columns(2)
                    
                    for i, val in enumerate(eigvals):
                        if i % 2 == 0:
                            with col_eig1:
                                st.markdown(f"λ{i+1} = **{val:.4f}**")
                        else:
                            with col_eig2:
                                st.markdown(f"λ{i+1} = **{val:.4f}**")
                    
                    st.markdown("**Eigenvectors (columns):**")
                    col_vec1, col_vec2 = st.columns(2)
                    
                    with col_vec1:
                        vec1 = eigvecs[:, 0]
                        st.latex(f"v_1 = \\begin{{bmatrix}}{vec1[0]:.4f} \\\\ {vec1[1]:.4f}\\end{{bmatrix}}")
                    
                    with col_vec2:
                        if eigvecs.shape[1] > 1:
                            vec2 = eigvecs[:, 1]
                            st.latex(f"v_2 = \\begin{{bmatrix}}{vec2[0]:.4f} \\\\ {vec2[1]:.4f}\\end{{bmatrix}}")
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    
                    if np.all(np.isreal(eigvals)):
                        if np.all(eigvals > 0):
                            stability = "All eigenvalues positive → expansion in all directions"
                        elif np.all(eigvals < 0):
                            stability = "All eigenvalues negative → contraction in all directions"
                        else:
                            stability = "Mixed signs → saddle point behavior"
                        
                        if np.abs(eigvals[0] - eigvals[1]) < 1e-10:
                            stability += " (equal eigenvalues → isotropic scaling)"
                        
                        st.info(stability)
                    else:
                        st.info("Complex eigenvalues indicate rotational behavior")
                    
                except Exception as e:
                    st.warning(f"Could not compute eigenvalues: {str(e)}")
            
        except Exception as e:
            st.error(f"Error creating eigen visualization: {str(e)}")

# ============ 3D DETERMINANT VISUALIZATION ============
if visualization_type == "3D Determinant Visualization":
    st.header("3D Determinant Visualization")
    st.markdown("""
    The determinant represents the scaling factor of volume in 3D.
    Blue wireframe: unit cube (volume = 1)
    Red wireframe: transformed cube (volume = |determinant|)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Matrix")
        st.markdown("Enter a 3×3 matrix:")
        
        # Handle examples
        if example_type == "Rotation":
            st.session_state.matrix_3d_det = [[0.866, -0.5, 0.0], [0.5, 0.866, 0.0], [0.0, 0.0, 1.0]]
        elif example_type == "Scaling":
            st.session_state.matrix_3d_det = [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.5]]
        elif example_type == "Shear":
            st.session_state.matrix_3d_det = [[1.0, 0.2, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]
        elif example_type == "Reflection":
            st.session_state.matrix_3d_det = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        A = create_matrix_table(3, 3, st.session_state.matrix_3d_det, key_suffix="3d_det")
        
        # Randomization button
        if st.button("🎲 Randomize Matrix", key="rand_3d_det"):
            st.session_state.matrix_3d_det = get_random_matrix(3, 3)
            st.session_state.randomize_trigger += 1
            st.rerun()
    
    with col2:
        st.subheader("3D Determinant Visualization")
        
        try:
            fig, det_value = visualize_3d_determinant(A)
            
            # Display 3D plot
            st.plotly_chart(
                fig, 
                use_container_width=True,
                height=700,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'displaylogo': False
                }
            )
            
            # 3D plot controls info
            with st.expander("3D View Controls"):
                st.markdown("""
                **Mouse Controls:**
                - **Rotate**: Click and drag
                - **Zoom**: Scroll wheel or right-click + drag
                - **Pan**: Right-click + drag or Ctrl + click + drag
                """)
            
            # Mathematical explanation in expander
            with st.expander("Mathematical Explanation"):
                st.markdown(f"""
                The determinant of a 3×3 matrix represents the volume scaling factor.
                
                **Calculation (Sarrus' rule):**
                $$
                \\det(A) = a_{{11}}a_{{22}}a_{{33}} + a_{{12}}a_{{23}}a_{{31}} + a_{{13}}a_{{21}}a_{{32}}
                         - a_{{31}}a_{{22}}a_{{13}} - a_{{32}}a_{{23}}a_{{11}} - a_{{33}}a_{{21}}a_{{12}}
                $$
                
                For this matrix:
                ```
                det = ({A[0][0]:.4f}×{A[1][1]:.4f}×{A[2][2]:.4f}) 
                    + ({A[0][1]:.4f}×{A[1][2]:.4f}×{A[2][0]:.4f}) 
                    + ({A[0][2]:.4f}×{A[1][0]:.4f}×{A[2][1]:.4f})
                    - ({A[2][0]:.4f}×{A[1][1]:.4f}×{A[0][2]:.4f})
                    - ({A[2][1]:.4f}×{A[1][2]:.4f}×{A[0][0]:.4f})
                    - ({A[2][2]:.4f}×{A[1][0]:.4f}×{A[0][1]:.4f})
                ```
                
                **Result:** det = **{det_value:.6f}**
                
                **Interpretation:**
                - **Volume scaling**: The transformed volume is **{abs(det_value):.4f}×** the original volume
                - **Orientation**: {"Positive (preserved)" if det_value >= 0 else "Negative (reversed)"}
                - **Invertibility**: {"Matrix is invertible" if abs(det_value) > 1e-10 else "Matrix is singular"}
                
                **Physical meaning:**
                The determinant tells you how much the transformation scales 3D volume.
                A determinant of 0 means the transformation collapses space into a lower dimension.
                """)
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

# ============ 3D EIGENVECTORS & EIGENVALUES ============
elif visualization_type == "3D Eigenvectors & Eigenvalues":
    st.header("3D Eigenvectors & Eigenvalues")
    st.markdown("""
    Visualize eigenvectors and eigenvalues in 3D space.
    
    **Visual elements explained:**
    - **Blue surface**: Unit sphere (original space)
    - **Red surface**: Transformed ellipsoid
    - **Solid colored arrows**: Eigenvectors (normalized)
    - **Dashed arrows**: Eigenvectors scaled by eigenvalues
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Matrix")
        st.markdown("Enter a 3×3 matrix:")
        
        # Handle examples
        if example_type == "Scaling":
            st.session_state.matrix_3d_eigen = [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.5]]
        elif example_type == "Rotation":
            st.session_state.matrix_3d_eigen = [[0.866, -0.5, 0.0], [0.5, 0.866, 0.0], [0.0, 0.0, 1.0]]
        elif example_type == "Shear":
            st.session_state.matrix_3d_eigen = [[1.0, 0.3, 0.1], [0.1, 1.0, 0.2], [0.0, 0.1, 1.0]]
        elif example_type == "Reflection":
            st.session_state.matrix_3d_eigen = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        A = create_matrix_table(3, 3, st.session_state.matrix_3d_eigen, key_suffix="3d_eigen")
        
        if st.button("🎲 Randomize Matrix", key="rand_3d_eigen"):
            st.session_state.matrix_3d_eigen = get_random_matrix(3, 3)
            st.session_state.randomize_trigger += 1
            st.rerun()
    
    with col2:
        st.subheader("3D Eigenvector Visualization")
        
        try:
            fig = visualize_3d_eigen(A)
            
            # Display 3D plot
            st.plotly_chart(
                fig, 
                use_container_width=True,
                height=700,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'displaylogo': False
                }
            )
            
            # Detailed explanation in expander
            with st.expander("Understanding 3D Eigen Analysis"):
                st.markdown("""
                **What you're seeing:**
                
                **1. Unit Sphere (Blue)**
                - Represents all unit vectors in the original space
                - Every point on this sphere has length 1
                
                **2. Transformed Ellipsoid (Red)**
                - Shows how the matrix transforms the unit sphere
                - The shape reveals the transformation's properties
                
                **3. Eigenvectors (Solid colored arrows)**
                - These are the principal axes of the ellipsoid
                - They point in directions that remain unchanged (except scaling)
                - Length shows direction (normalized to unit length)
                
                **4. Scaled Eigenvectors (Dashed arrows)**
                - Shows: eigenvector × eigenvalue
                - For a perfect eigenvector: points in same/opposite direction as solid arrow
                
                **Key Insights:**
                - The ellipsoid's axes align with eigenvectors
                - The ellipsoid's radii lengths are the absolute eigenvalues
                - Complex eigenvalues indicate rotational components
                """)
            
            # Mathematical analysis in expander
            with st.expander("Eigen Analysis Results"):
                try:
                    results = analyze_high_dim_matrix(A)
                    
                    # Display eigenvalues
                    st.markdown("### Eigenvalues")
                    col_eig1, col_eig2, col_eig3 = st.columns(3)
                    
                    eigvals = results['eigenvalues']
                    for i, val in enumerate(eigvals):
                        if i == 0:
                            with col_eig1:
                                if np.iscomplex(val):
                                    st.markdown(f"λ₁ = **{val.real:.4f} + {val.imag:.4f}i**")
                                else:
                                    st.markdown(f"λ₁ = **{val.real:.4f}**")
                        elif i == 1:
                            with col_eig2:
                                if np.iscomplex(val):
                                    st.markdown(f"λ₂ = **{val.real:.4f} + {val.imag:.4f}i**")
                                else:
                                    st.markdown(f"λ₂ = **{val.real:.4f}**")
                        elif i == 2:
                            with col_eig3:
                                if np.iscomplex(val):
                                    st.markdown(f"λ₃ = **{val.real:.4f} + {val.imag:.4f}i**")
                                else:
                                    st.markdown(f"λ₃ = **{val.real:.4f}**")
                    
                    # Display eigenvectors
                    st.markdown("### Eigenvectors (as columns)")
                    eigvecs = results['eigenvectors']
                    for i in range(3):
                        vec = eigvecs[i] if isinstance(eigvecs[0], list) else [row[i] for row in eigvecs]
                        st.latex(f"v_{i+1} = \\begin{{bmatrix}}{vec[0]:.4f} \\\\ {vec[1]:.4f} \\\\ {vec[2]:.4f}\\end{{bmatrix}}")
                    
                    # Matrix properties
                    st.markdown("### Matrix Properties")
                    col_prop1, col_prop2 = st.columns(2)
                    
                    with col_prop1:
                        st.metric("Determinant", f"{results['determinant']:.6f}")
                        st.metric("Trace", f"{results['trace']:.4f}")
                        st.metric("Rank", results['rank'])
                    
                    with col_prop2:
                        st.metric("Condition Number", f"{results['condition_number']:.4f}")
                        st.metric("Symmetric", "Yes" if results['is_symmetric'] else "No")
                        st.metric("Orthogonal", "Yes" if results['is_orthogonal'] else "No")
                    
                except Exception as e:
                    st.warning(f"Could not compute eigenvalues: {str(e)}")
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

# ============ HIGH-DIMENSIONAL CALCULATOR (4D-8D) ============
elif visualization_type == "High-Dimensional Calculator (4D-8D)":
    st.header("High-Dimensional Linear Algebra Calculator")
    st.markdown("""
    Analyze matrices and vectors in dimensions 4-8.
    This calculator provides detailed mathematical analysis without visualizations,
    focusing on computational results and interpretations.
    """)
    
    # Dimension selection
    st.subheader("1. Select Dimension")
    dimension = st.slider("Matrix Dimension (n×n)", 2, 8, st.session_state.high_dim_n, 
                         key="high_dim_slider")
    st.session_state.high_dim_n = dimension
    
    if "prev_high_dim_n" not in st.session_state:
        st.session_state.prev_high_dim_n = dimension

    if st.session_state.prev_high_dim_n != dimension:
        # Resize all stored vectors to match new dimension
        resized_vectors = []
        for vec in st.session_state.high_dim_vectors:
            new_vec = vec[:dimension] + [0.0] * (dimension - len(vec))
            resized_vectors.append(new_vec)
        st.session_state.high_dim_vectors = resized_vectors
        st.session_state.prev_high_dim_n = dimension
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("2. Input Matrix")
        st.markdown(f"Enter a {dimension}×{dimension} matrix:")
        
        # Ensure matrix has correct size
        current_size = len(st.session_state.high_dim_matrix)
        if current_size != dimension:
            # Resize matrix
            new_matrix = [[1.0 if i == j else 0.0 for j in range(dimension)] 
                         for i in range(dimension)]
            st.session_state.high_dim_matrix = new_matrix
        
        # Create matrix input
        A = []
        for i in range(dimension):
            cols = st.columns(dimension)
            row_vals = []
            for j in range(dimension):
                with cols[j]:
                    key = f"high_dim_mat_{i}_{j}"
                    val = st.number_input(
                        f"[{i},{j}]",
                        value=float(st.session_state.high_dim_matrix[i][j]),
                        step=0.1,
                        format="%.4f",
                        key=key,
                        label_visibility="collapsed"
                    )
                    row_vals.append(val)
            A.append(row_vals)
        
        # Update session state
        st.session_state.high_dim_matrix = A
        
        # Matrix operations
        st.markdown("**Matrix Operations**")
        op_col1, op_col2, op_col3 = st.columns(3)
        
        with op_col1:
            if st.button("🎲 Randomize Matrix"):
                st.session_state.high_dim_matrix = get_random_matrix(dimension, dimension, scale=3)
                st.rerun()
        
        with op_col2:
            if st.button("🔄 Reset to Identity"):
                st.session_state.high_dim_matrix = [[1.0 if i == j else 0.0 for j in range(dimension)] 
                                                   for i in range(dimension)]
                st.rerun()
        
        with op_col3:
            if st.button("🔄 Random Symmetric"):
                # Generate random symmetric matrix
                B = get_random_matrix(dimension, dimension, scale=2)
                sym_matrix = [[(B[i][j] + B[j][i])/2 for j in range(dimension)] 
                             for i in range(dimension)]
                st.session_state.high_dim_matrix = sym_matrix
                st.rerun()
        
        # Show matrix in LaTeX
        with st.expander("View Matrix in LaTeX"):
            latex_str = "A = \\begin{bmatrix}\n"
            for i in range(dimension):
                row_str = " & ".join([f"{val:.4f}" for val in A[i]])
                latex_str += row_str
                if i < dimension - 1:
                    latex_str += " \\\\\n"
            latex_str += "\n\\end{bmatrix}"
            st.latex(latex_str)
    
    with col2:
        st.subheader("3. Vector Transformation")
        st.markdown(f"Enter {dimension}-dimensional vectors to transform:")
        
        num_vectors = st.slider("Number of vectors", 1, 5, 
                               min(2, dimension), key="high_dim_num_vec")
        
        # Ensure vectors have correct size
        if len(st.session_state.high_dim_vectors) != num_vectors:
            # Initialize basis vectors
            new_vectors = []
            for i in range(num_vectors):
                if i < dimension:
                    vec = [1.0 if j == i else 0.0 for j in range(dimension)]
                else:
                    vec = [1.0 for _ in range(dimension)]
                new_vectors.append(vec)
            st.session_state.high_dim_vectors = new_vectors
        
        for k in range(len(st.session_state.high_dim_vectors)):
            vec = st.session_state.high_dim_vectors[k]
            if len(vec) != dimension:
                st.session_state.high_dim_vectors[k] = vec[:dimension] + [0.0] * (dimension - len(vec))
        
        vectors = []
        for vec_idx in range(num_vectors):
            st.markdown(f"**Vector {vec_idx + 1}:**")
            cols = st.columns(dimension)
            vec_vals = []
            for i in range(dimension):
                with cols[i]:
                    key = f"high_dim_vec_{vec_idx}_{i}"
                    val = st.number_input(
                        f"[{i}]",
                        value=float(st.session_state.high_dim_vectors[vec_idx][i]),
                        step=0.1,
                        format="%.4f",
                        key=key,
                        label_visibility="collapsed"
                    )
                    vec_vals.append(val)
            vectors.append(vec_vals)
            st.session_state.high_dim_vectors[vec_idx] = vec_vals
        
        if st.button("🎲 Randomize Vectors"):
            st.session_state.high_dim_vectors = get_random_vectors(num_vectors, dimension, scale=3)
            st.rerun()
    
    # Analysis Results
    st.subheader("4. Analysis Results")
    
    try:
        # Analyze matrix
        analysis = analyze_high_dim_matrix(A)
        matrix_props = compute_matrix_properties(A)
        inverse_result = compute_matrix_inverse(A)
        
        # Transform vectors
        if vectors:
            transformation = transform_vectors(A, vectors)
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Matrix Properties", 
            "Eigen Analysis", 
            "Matrix Inverse",
            "Vector Transformation", 
            "Interpretation"
        ])
        
        with tab1:
            st.markdown("### Matrix Properties")
            
            # Basic properties
            col_prop1, col_prop2, col_prop3 = st.columns(3)
            
            with col_prop1:
                st.metric("Determinant", f"{analysis['determinant']:.6f}")
                st.metric("Trace", f"{analysis['trace']:.4f}")
                st.metric("Rank", analysis['rank'])
                st.metric("Condition Number", f"{analysis['condition_number']:.4f}")
            
            with col_prop2:
                st.metric("Frobenius Norm", f"{matrix_props['frobenius_norm']:.4f}")
                st.metric("Spectral Norm", f"{matrix_props['spectral_norm']:.4f}")
                st.metric("Spectral Radius", f"{matrix_props.get('spectral_radius', 'N/A'):.4f}")
                st.metric("Dimension", analysis['dimension'])
            
            with col_prop3:
                # Matrix type indicators
                type_indicators = [
                    ("Symmetric", matrix_props['is_symmetric']),
                    ("Orthogonal", matrix_props['is_orthogonal']),
                    ("Diagonal", matrix_props['is_diagonal']),
                    ("Identity", matrix_props['is_identity']),
                ]
                
                for name, value in type_indicators:
                    if value:
                        st.success(f"✓ {name}")
                    else:
                        st.info(f"○ {name}")
            
            # Matrix type details
            with st.expander("Matrix Type Details"):
                col_type1, col_type2 = st.columns(2)
                with col_type1:
                    st.write("**Structural Properties:**")
                    type_details = [
                        ("Upper Triangular", matrix_props['is_upper_triangular']),
                        ("Lower Triangular", matrix_props['is_lower_triangular']),
                        ("Skew-Symmetric", matrix_props['is_skew_symmetric']),
                    ]
                    for name, value in type_details:
                        status = "Yes" if value else "No"
                        st.write(f"{name}: {status}")
                
                with col_type2:
                    st.write("**Numerical Properties:**")
                    st.write(f"Condition Number: {analysis['condition_number']:.4f}")
                    cond_status = "Well-conditioned" if analysis['condition_number'] < 1e6 else "Ill-conditioned"
                    st.write(f"Condition: {cond_status}")
                    st.write(f"Singular: {'Yes' if abs(analysis['determinant']) < 1e-10 else 'No'}")
            
            # Singular Value Decomposition for additional insight
            if dimension <= 6:
                try:
                    U, S, Vh = np.linalg.svd(A)
                    st.markdown("**Singular Values:**")
                    sv_df = pd.DataFrame({
                        'Index': range(1, len(S) + 1),
                        'Value': S,
                        'Percentage': S / S.sum() * 100
                    })
                    st.dataframe(sv_df.style.format({'Value': '{:.4f}', 'Percentage': '{:.2f}%'}))
                except:
                    pass
        
        with tab2:
            st.markdown("### Eigenvalues and Eigenvectors")
            
            if analysis['eigenvalues'] is not None:
                eigvals = analysis['eigenvalues']
                
                # Display eigenvalues
                st.markdown("**Eigenvalues:**")
                cols = st.columns(min(4, dimension))
                for i, val in enumerate(eigvals):
                    col_idx = i % 4
                    if i % 4 == 0 and i > 0:
                        cols = st.columns(min(4, dimension - i))
                    
                    with cols[col_idx]:
                        if np.iscomplex(val):
                            st.markdown(f"λ{i+1} = {val.real:.4f} + {val.imag:.4f}i")
                        else:
                            st.markdown(f"λ{i+1} = {val.real:.4f}")
                
                # Display eigenvectors if not too large
                if dimension <= 6:
                    st.markdown("**Eigenvectors (columns of V):**")
                    eigvecs = analysis['eigenvectors']
                    for i in range(dimension):
                        if isinstance(eigvecs[0], list):
                            vec = [row[i] for row in eigvecs]
                        else:
                            vec = eigvecs[i]
                        
                        vec_str = "\\\\".join([f"{v:.4f}" for v in vec])
                        st.latex(f"v_{i+1} = \\begin{{bmatrix}}{vec_str}\\end{{bmatrix}}")
                
                # Eigenvalue properties
                st.markdown("**Eigenvalue Analysis:**")
                real_parts = analysis['eigenvalues_real']
                
                col_eig1, col_eig2, col_eig3, col_eig4 = st.columns(4)
                with col_eig1:
                    max_real = max(real_parts)
                    st.metric("Max Real Part", f"{max_real:.4f}")
                with col_eig2:
                    min_real = min(real_parts)
                    st.metric("Min Real Part", f"{min_real:.4f}")
                with col_eig3:
                    has_complex = analysis['has_complex_eigenvalues']
                    st.metric("Complex Eigenvalues", "Yes" if has_complex else "No")
                with col_eig4:
                    if 'spectral_radius' in matrix_props:
                        st.metric("Spectral Radius", f"{matrix_props['spectral_radius']:.4f}")
        
        with tab3:
            st.markdown("### Matrix Inverse")
            
            if inverse_result['exists']:
                st.success("✅ Matrix is invertible")
                
                # Show inverse matrix
                st.markdown("**Inverse Matrix A⁻¹:**")
                
                if dimension <= 6:
                    # Display as LaTeX
                    inv_matrix = inverse_result['inverse']
                    latex_str = "A^{-1} = \\begin{bmatrix}\n"
                    for i in range(dimension):
                        row_str = " & ".join([f"{val:.6f}" for val in inv_matrix[i]])
                        latex_str += row_str
                        if i < dimension - 1:
                            latex_str += " \\\\\n"
                    latex_str += "\n\\end{bmatrix}"
                    st.latex(latex_str)
                else:
                    st.info(f"Inverse computed successfully. Matrix is {dimension}×{dimension}.")
                    with st.expander("View Inverse Matrix"):
                        inv_matrix = inverse_result['inverse']
                        for i in range(dimension):
                            row_str = "  ".join([f"{val:10.6f}" for val in inv_matrix[i]])
                            st.text(row_str)
                
                # Inverse properties
                st.markdown("**Inverse Properties:**")
                
                col_inv1, col_inv2, col_inv3 = st.columns(3)
                with col_inv1:
                    st.metric("Determinant", f"{inverse_result['determinant']:.6e}")
                    st.metric("Product Error", f"{inverse_result['product_error']:.2e}")
                
                with col_inv2:
                    cond_status = "Good" if inverse_result['is_well_conditioned'] else "Poor"
                    st.metric("Condition", cond_status)
                    st.metric("Condition Number", f"{inverse_result['condition_number']:.2e}")
                
                with col_inv3:
                    accuracy = "High" if inverse_result['is_exact'] else "Moderate"
                    st.metric("Numerical Accuracy", accuracy)
                    st.metric("Inverse Exists", "Yes")
                
                # Test inverse properties
                with st.expander("Verify Inverse Properties"):
                    A_np = np.array(A)
                    inv_np = np.array(inverse_result['inverse'])
                    
                    # Check A * A⁻¹ ≈ I
                    product = A_np @ inv_np
                    identity = np.eye(dimension)
                    error = np.linalg.norm(product - identity)
                    
                    st.write(f"**A × A⁻¹ ≈ I:** Error = {error:.2e}")
                    if error < 1e-10:
                        st.success("✓ Inverse verification passed")
                    else:
                        st.warning(f"⚠ Small numerical error: {error:.2e}")
                    
                    # Check A⁻¹ * A ≈ I
                    product2 = inv_np @ A_np
                    error2 = np.linalg.norm(product2 - identity)
                    st.write(f"**A⁻¹ × A ≈ I:** Error = {error2:.2e}")
                    
                    # Determinant relationship: det(A⁻¹) = 1/det(A)
                    det_A = np.linalg.det(A_np)
                    det_inv = np.linalg.det(inv_np)
                    expected = 1 / det_A
                    rel_error = abs(det_inv - expected) / abs(expected)
                    st.write(f"**det(A⁻¹) = 1/det(A):** Relative error = {rel_error:.2e}")
                    
            else:
                st.error("❌ Matrix is not invertible")
                st.info(f"**Reason:** {inverse_result['error_message']}")
                
                # Suggest alternatives
                st.markdown("**Alternatives for Non-Invertible Matrices:**")
                st.markdown("""
                1. **Pseudoinverse (Moore-Penrose):** Use `np.linalg.pinv()` for least-squares solutions
                2. **Regularization:** Add small value to diagonal: A + εI
                3. **SVD-based inversion:** Use singular value decomposition with thresholding
                """)
                
                # Option to compute pseudoinverse
                if st.button("Compute Pseudoinverse (Moore-Penrose)"):
                    try:
                        pinv = np.linalg.pinv(A)
                        st.success("Pseudoinverse computed successfully")
                        
                        # Display pseudoinverse
                        if dimension <= 6:
                            st.markdown("**Pseudoinverse A⁺:**")
                            latex_str = "A^{+} = \\begin{bmatrix}\n"
                            for i in range(dimension):
                                row_str = " & ".join([f"{val:.6f}" for val in pinv[i]])
                                latex_str += row_str
                                if i < dimension - 1:
                                    latex_str += " \\\\\n"
                            latex_str += "\n\\end{bmatrix}"
                            st.latex(latex_str)
                    except Exception as e:
                        st.error(f"Error computing pseudoinverse: {str(e)}")
        
        with tab4:
            if vectors:
                st.markdown("### Vector Transformation Results")
                
                for i in range(len(vectors)):
                    st.markdown(f"**Vector {i+1}:**")
                    
                    col_vec1, col_vec2, col_vec3 = st.columns(3)
                    
                    with col_vec1:
                        st.markdown("**Original:**")
                        orig_vec = transformation['original_vectors'][i]
                        orig_str = "\\\\".join([f"{v:.4f}" for v in orig_vec])
                        st.latex(f"v = \\begin{{bmatrix}}{orig_str}\\end{{bmatrix}}")
                        st.metric("Norm", f"{transformation['original_norms'][i]:.4f}")
                    
                    with col_vec2:
                        st.markdown("**Transformed:**")
                        trans_vec = transformation['transformed_vectors'][i]
                        trans_str = "\\\\".join([f"{v:.4f}" for v in trans_vec])
                        st.latex(f"Av = \\begin{{bmatrix}}{trans_str}\\end{{bmatrix}}")
                        st.metric("Norm", f"{transformation['transformed_norms'][i]:.4f}")
                    
                    with col_vec3:
                        # Transformation metrics
                        scaling = transformation['scaling_factors'][i]
                        angle = transformation['angles_degrees'][i]
                        st.metric("Scaling Factor", f"{scaling:.4f}")
                        st.metric("Angle Change", f"{angle:.2f}°")
                    
                    # If inverse exists, show reverse transformation
                    if inverse_result['exists']:
                        st.markdown("**Reverse Transformation (using inverse):**")
                        try:
                            inv_matrix = inverse_result['inverse']
                            # Transform back: A⁻¹ * (A*v) should ≈ v
                            trans_vec_np = np.array(trans_vec)
                            inv_np = np.array(inv_matrix)
                            recovered = inv_np @ trans_vec_np
                            
                            col_rec1, col_rec2 = st.columns(2)
                            with col_rec1:
                                rec_str = "\\\\".join([f"{v:.6f}" for v in recovered])
                                st.latex(f"A^{{-1}}(Av) = \\begin{{bmatrix}}{rec_str}\\end{{bmatrix}}")
                            
                            with col_rec2:
                                error = np.linalg.norm(recovered - orig_vec)
                                st.metric("Recovery Error", f"{error:.2e}")
                        except:
                            pass
                    
                    st.markdown("---")
        
        with tab5:
            st.markdown("### Mathematical Interpretation")
            
            # Stability analysis for control theory
            if dimension <= 4:
                st.markdown("**Stability Analysis (for ẋ = Ax):**")
                
                if analysis['is_stable']:
                    st.success("✅ **Stable System**: All eigenvalues have negative real parts.")
                    st.markdown("""
                    - Solutions decay to zero over time
                    - System returns to equilibrium after perturbations
                    """)
                elif analysis['marginally_stable']:
                    st.warning("⚠️ **Marginally Stable**: Some eigenvalues have zero real parts.")
                    st.markdown("""
                    - Solutions may oscillate or remain bounded
                    - System doesn't decay to zero but doesn't grow either
                    """)
                elif analysis['unstable']:
                    st.error("❌ **Unstable System**: At least one eigenvalue has positive real part.")
                    st.markdown("""
                    - Solutions grow without bound
                    - System diverges from equilibrium
                    """)
            
            # Determinant interpretation
            det = analysis['determinant']
            st.markdown("**Determinant Interpretation:**")
            if abs(det) < 1e-10:
                st.info("**Singular Matrix**: Determinant ≈ 0")
                st.markdown("""
                - The transformation collapses space to lower dimension
                - Matrix is not invertible
                - Columns are linearly dependent
                """)
            else:
                st.info(f"**Volume Scaling**: |det| = {abs(det):.4f}")
                st.markdown(f"""
                - Transformed volume is **{abs(det):.4f}×** original volume
                - {"Preserves orientation" if det > 0 else "Reverses orientation"}
                - Matrix is invertible
                """)
            
            # Inverse interpretation
            if inverse_result['exists']:
                st.markdown("**Inverse Matrix Interpretation:**")
                st.markdown(f"""
                - A⁻¹ undoes the transformation of A: A⁻¹(Av) = v for any vector v
                - Condition number: {inverse_result['condition_number']:.2e} ({'well-conditioned' if inverse_result['is_well_conditioned'] else 'ill-conditioned'})
                - Numerical accuracy: {'High' if inverse_result['is_exact'] else 'Moderate'}
                """)
            
            # Eigenvalue interpretation
            if analysis['eigenvalues'] is not None:
                eigvals = analysis['eigenvalues_real']
                has_complex = analysis['has_complex_eigenvalues']
                
                st.markdown("**Eigenvalue Interpretation:**")
                if has_complex:
                    st.info("Complex eigenvalues indicate rotational components in the transformation.")
                if all(abs(val - 1) < 0.1 for val in eigvals):
                    st.info("Eigenvalues near 1 suggest the transformation is close to identity.")
                if any(abs(val) < 0.1 for val in eigvals):
                    st.info("Small eigenvalues indicate directions of strong compression.")
                if all(val > 0 for val in eigvals):
                    st.info("All eigenvalues positive: transformation expands in all eigen-directions.")
                if all(val < 0 for val in eigvals):
                    st.info("All eigenvalues negative: transformation contracts in all eigen-directions.")
        
# ============ EXPORT/IMPORT SECTION ============
        st.markdown("---")
        st.subheader("📤 Export/Import Data")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Prepare analysis data with proper Python types for JSON serialization
            det_value = analysis.get('determinant')
            if det_value is not None:
                det_value = float(det_value)
            
            eig_values = analysis.get('eigenvalues_real', [])
            if eig_values is not None:
                eig_values = [float(val) for val in eig_values]
            
            rank_value = analysis.get('rank')
            if rank_value is not None:
                rank_value = int(rank_value)
            
            cond_value = analysis.get('condition_number')
            if cond_value is not None:
                cond_value = float(cond_value)
            
            # Export as JSON
            export_data = {
                "matrix": A,
                "vectors": vectors if vectors else [],
                "dimension": dimension,
                "timestamp": datetime.datetime.now().isoformat(),
                "analysis": {
                    "determinant": det_value,
                    "eigenvalues": eig_values,
                    "rank": rank_value,
                    "condition_number": cond_value
                }
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Export as JSON",
                data=json_str,
                file_name=f"matrix_analysis_{dimension}d_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_json"
            )
        
        with export_col2:
            # Export as Python code
            python_code = f'''
# Generated by Linear Algebra Visualizer
import numpy as np

# Matrix A ({dimension}×{dimension})
A = np.array({str(A)})

# Vectors to transform
vectors = {str(vectors)}

# Analysis results
determinant = {det_value if det_value is not None else 0:.6f}
rank = {rank_value if rank_value is not None else 0}
condition_number = {cond_value if cond_value is not None else 0:.4f}

print(f"Matrix A:\\n{{A}}\\n")
print(f"Determinant: {{determinant}}")
print(f"Rank: {{rank}}")
print(f"Condition Number: {{condition_number}}")
'''
            
            st.download_button(
                label="🐍 Export as Python",
                data=python_code,
                file_name=f"matrix_code_{dimension}d.py",
                mime="text/plain",
                key="export_python"
            )
        
        with export_col3:
            # Import from JSON
            uploaded_file = st.file_uploader("📂 Import JSON", type=['json'], key="import_json")
            if uploaded_file is not None:
                try:
                    import_data = json.load(uploaded_file)
                    if 'matrix' in import_data and 'dimension' in import_data:
                        if import_data['dimension'] == dimension:
                            st.session_state.high_dim_matrix = import_data['matrix']
                            if 'vectors' in import_data:
                                st.session_state.high_dim_vectors = import_data['vectors']
                            st.success(f"✅ Imported {dimension}×{dimension} matrix from {uploaded_file.name}")
                            st.rerun()
                        else:
                            st.warning(f"Imported dimension {import_data['dimension']} doesn't match current dimension {dimension}")
                except Exception as e:
                    st.error(f"Error importing file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.info("Please check your matrix and vector inputs.")    
    

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
### How to Use This Tool

**Visualizations (2D/3D):**
1. Select a visualization from the sidebar
2. Enter matrix values using the table inputs
3. Adjust parameters as needed
4. Click expanders for detailed mathematical explanations
5. Interact with plots (rotate, zoom, pan)

**High-Dimensional Calculator:**
1. Select dimension (2-8)
2. Enter matrix and vectors
3. Explore analysis in tabs:
   - Eigen Analysis: Eigenvalues and eigenvectors
   - Matrix Properties: Determinant, trace, rank, etc.
   - Vector Transformation: How vectors change
   - Interpretation: Mathematical meaning and control theory insights

**Randomization:** Click the 🎲 buttons to generate random matrices/vectors
**Note:** Inputs automatically update when you change values
""")