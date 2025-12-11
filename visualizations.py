"""
Linear Algebra Visualization Functions
Core mathematical operations and visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Union
import math

# ============ UTILITY FUNCTIONS ============

def validate_matrix(matrix: List[List[float]]) -> np.ndarray:
    """Convert and validate matrix input"""
    arr = np.array(matrix, dtype=float)
    if len(arr.shape) != 2:
        raise ValueError("Matrix must be 2D")
    return arr

def validate_vector(vector: List[float]) -> np.ndarray:
    """Convert and validate vector input"""
    arr = np.array(vector, dtype=float)
    if len(arr.shape) != 1:
        raise ValueError("Vector must be 1D")
    return arr

# ============ 2D VISUALIZATIONS ============

def visualize_2d_transformation(
    matrix: List[List[float]], 
    vectors: List[List[float]] = None
) -> go.Figure:
    """
    Visualize 2D vector transformations
    
    Args:
        matrix: 2x2 transformation matrix
        vectors: List of 2D vectors to transform
    
    Returns:
        Plotly figure object
    """
    A = validate_matrix(matrix)
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for 2D transformations")
    
    # Default vectors if none provided
    if vectors is None:
        vectors = [[1, 0], [0, 1], [1, 1]]  # Standard basis + diagonal
    
    # Prepare data
    original_vectors = [validate_vector(v) for v in vectors]
    transformed_vectors = [A @ v for v in original_vectors]
    
    # Create figure
    fig = go.Figure()
    
    # Plot original vectors
    for i, v in enumerate(original_vectors):
        fig.add_trace(go.Scatter(
            x=[0, v[0]],
            y=[0, v[1]],
            mode='lines+markers',
            name=f'Original v{i+1}',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
    
    # Plot transformed vectors
    for i, v in enumerate(transformed_vectors):
        fig.add_trace(go.Scatter(
            x=[0, v[0]],
            y=[0, v[1]],
            mode='lines+markers',
            name=f'Transformed v{i+1}',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10)
        ))
    
    # Add unit circle/grid for reference
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        name='Unit Circle',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=True
    ))
    
    # Style the plot
    fig.update_layout(
        title="2D Vector Transformation",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            gridcolor='lightgray',
            zerolinecolor='black'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='black'
        ),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def visualize_2d_determinant(
    matrix: List[List[float]]
) -> Tuple[go.Figure, float]:
    """
    Visualize determinant as area scaling in 2D
    
    Args:
        matrix: 2x2 matrix
    
    Returns:
        (figure, determinant_value)
    """
    A = validate_matrix(matrix)
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for 2D determinant")
    
    det = np.linalg.det(A)
    
    # Unit square vertices
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    
    # Transformed square
    transformed_square = (A @ unit_square.T).T
    
    fig = go.Figure()
    
    # Unit square
    fig.add_trace(go.Scatter(
        x=unit_square[:, 0], y=unit_square[:, 1],
        mode='lines+markers',
        name='Unit Square (Area = 1)',
        line=dict(color='blue', width=2),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    # Transformed square
    fig.add_trace(go.Scatter(
        x=transformed_square[:, 0], y=transformed_square[:, 1],
        mode='lines+markers',
        name=f'Transformed (Area = {det:.2f})',
        line=dict(color='red', width=2),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    # Add eigenvectors if they exist
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        for i in range(2):
            vec = eigvecs[:, i]
            fig.add_trace(go.Scatter(
                x=[0, vec[0].real], y=[0, vec[1].real],
                mode='lines',
                name=f'Eigenvector {i+1} (λ={eigvals[i]:.2f})',
                line=dict(color='green', width=3)
            ))
    except:
        pass
    
    fig.update_layout(
        title=f"2D Determinant Visualization (det = {det:.2f})",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig, det

# ============ 3D VISUALIZATIONS ============

def visualize_3d_transformation(
    matrix: List[List[float]],
    vectors: List[List[float]] = None
) -> go.Figure:
    """
    Visualize 3D vector transformations
    
    Args:
        matrix: 3x3 transformation matrix
        vectors: List of 3D vectors to transform
    
    Returns:
        Plotly figure object
    """
    A = validate_matrix(matrix)
    if A.shape != (3, 3):
        raise ValueError("Matrix must be 3x3 for 3D transformations")
    
    # Default vectors (standard basis)
    if vectors is None:
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    original_vectors = [validate_vector(v) for v in vectors]
    transformed_vectors = [A @ v for v in original_vectors]
    
    fig = go.Figure()
    
    colors = ['blue', 'green', 'orange']
    
    # Plot original vectors
    for i, v in enumerate(original_vectors):
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode='lines+markers',
            name=f'Original e{i+1}',
            line=dict(color=colors[i], width=5),
            marker=dict(size=4)
        ))
    
    # Plot transformed vectors
    for i, v in enumerate(transformed_vectors):
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode='lines+markers',
            name=f'Transformed e{i+1}',
            line=dict(color=colors[i], width=5, dash='dash'),
            marker=dict(size=4)
        ))
    
    # Add unit sphere for reference (wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.1,
        colorscale='gray',
        showscale=False,
        name='Unit Sphere'
    ))
    
    fig.update_layout(
        title="3D Vector Transformation",
        scene=dict(
            aspectmode='cube',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        showlegend=True
    )
    
    return fig

# ============ EIGEN VISUALIZATIONS ============

def visualize_eigen_2d(
    matrix: List[List[float]]
) -> go.Figure:
    """
    Visualize eigenvectors and eigenvalues in 2D
    
    Args:
        matrix: 2x2 matrix
    
    Returns:
        Plotly figure with eigenvectors
    """
    A = validate_matrix(matrix)
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for eigen visualization")
    
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(A)
    
    fig = go.Figure()
    
    # Plot a grid of points
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Original points
    for point in points[::5]:  # Sample for clarity
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]],
            mode='markers',
            marker=dict(color='blue', size=5, opacity=0.5),
            showlegend=False
        ))
    
    # Transformed points
    transformed_points = (A @ points.T).T
    for i, point in enumerate(points[::5]):
        fig.add_trace(go.Scatter(
            x=[transformed_points[i, 0]], y=[transformed_points[i, 1]],
            mode='markers',
            marker=dict(color='red', size=5, opacity=0.5),
            showlegend=False
        ))
        # Draw lines from original to transformed
        fig.add_trace(go.Scatter(
            x=[point[0], transformed_points[i, 0]],
            y=[point[1], transformed_points[i, 1]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
    
    # Plot eigenvectors
    for i in range(2):
        vec = eigvecs[:, i]
        # Scale eigenvector by eigenvalue
        scaled_vec = eigvals[i] * vec
        
        fig.add_trace(go.Scatter(
            x=[0, vec[0].real], y=[0, vec[1].real],
            mode='lines+markers',
            name=f'Eigenvector {i+1} (λ={eigvals[i]:.2f})',
            line=dict(color='green', width=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, scaled_vec[0].real], y=[0, scaled_vec[1].real],
            mode='lines+markers',
            name=f'λ * v{i+1}',
            line=dict(color='purple', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"2D Eigenvectors (λ₁={eigvals[0]:.2f}, λ₂={eigvals[1]:.2f})",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    
    return fig

# ============ 3D DETERMINANT VISUALIZATION ============

def visualize_3d_determinant(
    matrix: List[List[float]]
) -> Tuple[go.Figure, float]:
    """
    Visualize determinant as volume scaling in 3D
    
    Args:
        matrix: 3x3 matrix
    
    Returns:
        (figure, determinant_value)
    """
    A = validate_matrix(matrix)
    if A.shape != (3, 3):
        raise ValueError("Matrix must be 3x3 for 3D determinant")
    
    det = np.linalg.det(A)
    
    # Unit cube vertices
    unit_cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Transform cube vertices
    transformed_vertices = (A @ unit_cube_vertices.T).T
    
    # Define cube faces (vertex indices)
    faces = [
        [0, 1, 2, 3],  # Bottom
        [4, 5, 6, 7],  # Top
        [0, 1, 5, 4],  # Front
        [2, 3, 7, 6],  # Back
        [1, 2, 6, 5],  # Right
        [0, 3, 7, 4]   # Left
    ]
    
    fig = go.Figure()
    
    # Plot unit cube (wireframe)
    for face in faces:
        x = unit_cube_vertices[face + [face[0]], 0]
        y = unit_cube_vertices[face + [face[0]], 1]
        z = unit_cube_vertices[face + [face[0]], 2]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name='Unit Cube (V=1)',
            line=dict(color='blue', width=2),
            showlegend=True if face == faces[0] else False
        ))
    
    # Plot transformed cube (wireframe)
    for face in faces:
        x = transformed_vertices[face + [face[0]], 0]
        y = transformed_vertices[face + [face[0]], 1]
        z = transformed_vertices[face + [face[0]], 2]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name=f'Transformed (V={abs(det):.2f})',
            line=dict(color='red', width=2),
            showlegend=True if face == faces[0] else False
        ))
    
    # Add eigenvectors if they exist
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        for i in range(3):
            vec = eigvecs[:, i]
            # Normalize for visualization
            vec_norm = vec / np.linalg.norm(vec) * 1.5
            
            fig.add_trace(go.Scatter3d(
                x=[0, vec_norm[0].real], 
                y=[0, vec_norm[1].real], 
                z=[0, vec_norm[2].real],
                mode='lines',
                name=f'Eigenvector {i+1} (λ={eigvals[i]:.2f})',
                line=dict(color='green', width=4)
            ))
    except:
        pass
    
    fig.update_layout(
        title=f"3D Determinant Visualization (det = {det:.4f})",
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )
    
    return fig, det

# ============ 3D EIGENVECTORS VISUALIZATION ============

def visualize_3d_eigen(
    matrix: List[List[float]]
) -> go.Figure:
    """
    Visualize eigenvectors and eigenvalues in 3D
    
    Args:
        matrix: 3x3 matrix
    
    Returns:
        Plotly figure with eigenvectors
    """
    A = validate_matrix(matrix)
    if A.shape != (3, 3):
        raise ValueError("Matrix must be 3x3 for 3D eigen visualization")
    
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Create unit sphere for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Transform sphere to show deformation
    points = np.vstack([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()]).T
    transformed_points = (A @ points.T).T
    x_trans = transformed_points[:, 0].reshape(x_sphere.shape)
    y_trans = transformed_points[:, 1].reshape(y_sphere.shape)
    z_trans = transformed_points[:, 2].reshape(z_sphere.shape)
    
    fig = go.Figure()
    
    # Original unit sphere (wireframe)
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        colorscale='Blues',
        showscale=False,
        name='Unit Sphere',
        showlegend=True
    ))
    
    # Transformed ellipsoid (wireframe)
    fig.add_trace(go.Surface(
        x=x_trans, y=y_trans, z=z_trans,
        opacity=0.1,
        colorscale='Reds',
        showscale=False,
        name='Transformed Ellipsoid',
        showlegend=True
    ))
    
    # Plot eigenvectors
    colors = ['green', 'orange', 'purple']
    for i in range(3):
        vec = eigvecs[:, i]
        # Normalize eigenvector for visualization
        if np.linalg.norm(vec) > 0:
            vec_norm = vec / np.linalg.norm(vec)
        else:
            vec_norm = vec
        
        # Plot eigenvector
        fig.add_trace(go.Scatter3d(
            x=[0, vec_norm[0].real * 2], 
            y=[0, vec_norm[1].real * 2], 
            z=[0, vec_norm[2].real * 2],
            mode='lines+markers',
            name=f'Eigenvector {i+1} (λ={eigvals[i]:.4f})',
            line=dict(color=colors[i], width=6),
            marker=dict(size=4)
        ))
        
        # Plot scaled eigenvector (eigenvalue * eigenvector)
        scaled_vec = eigvals[i] * vec_norm * 2
        fig.add_trace(go.Scatter3d(
            x=[0, scaled_vec[0].real], 
            y=[0, scaled_vec[1].real], 
            z=[0, scaled_vec[2].real],
            mode='lines',
            name=f'λ{i+1} * v{i+1}',
            line=dict(color=colors[i], width=3, dash='dash'),
            showlegend=True if i == 0 else False
        ))
    
    fig.update_layout(
        title=f"3D Eigenvectors (λ₁={eigvals[0]:.4f}, λ₂={eigvals[1]:.4f}, λ₃={eigvals[2]:.4f})",
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )
    
    return fig

# ============ HIGH-DIMENSIONAL CALCULATOR FUNCTIONS ============

def analyze_high_dim_matrix(matrix: List[List[float]]) -> dict:
    """
    Analyze an n×n matrix (n ≤ 8) and return comprehensive results
    
    Args:
        matrix: n×n matrix
    
    Returns:
        Dictionary with analysis results
    """
    A = validate_matrix(matrix)
    n = A.shape[0]
    
    if n > 8:
        raise ValueError(f"Matrix dimension {n} exceeds maximum allowed (8)")
    
    results = {
        'dimension': n,
        'matrix': A.tolist(),
        'determinant': np.linalg.det(A),
        'trace': np.trace(A),
        'rank': np.linalg.matrix_rank(A),
        'condition_number': np.linalg.cond(A),
        'is_symmetric': np.allclose(A, A.T),
        'is_orthogonal': np.allclose(A @ A.T, np.eye(n)),
        'is_diagonal': np.allclose(A, np.diag(np.diag(A))),
    }
    
    # Eigen analysis
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        results['eigenvalues'] = eigvals.tolist()
        results['eigenvectors'] = eigvecs.tolist()
        results['has_complex_eigenvalues'] = np.any(np.iscomplex(eigvals))
        
        # Real and imaginary parts
        results['eigenvalues_real'] = eigvals.real.tolist()
        results['eigenvalues_imag'] = eigvals.imag.tolist()
        
        # Stability analysis for control theory
        if n <= 4:  # Only for smaller matrices
            real_parts = eigvals.real
            results['is_stable'] = np.all(real_parts < 0)
            results['marginally_stable'] = np.all(real_parts <= 0) and np.any(real_parts == 0)
            results['unstable'] = np.any(real_parts > 0)
    except:
        results['eigenvalues'] = None
        results['eigenvectors'] = None
    
    return results

def transform_vectors(matrix: List[List[float]], vectors: List[List[float]]) -> dict:
    """
    Transform vectors using the matrix
    
    Args:
        matrix: n×n matrix
        vectors: List of n-dimensional vectors
    
    Returns:
        Dictionary with transformation results
    """
    A = validate_matrix(matrix)
    n = A.shape[0]
    
    if any(len(v) != n for v in vectors):
        raise ValueError(f"All vectors must have dimension {n}")
    
    vectors_np = np.array(vectors, dtype=float)
    transformed = (A @ vectors_np.T).T
    
    # Calculate norms and angles
    original_norms = np.linalg.norm(vectors_np, axis=1)
    transformed_norms = np.linalg.norm(transformed, axis=1)
    scaling_factors = transformed_norms / original_norms
    
    # Angles between original and transformed vectors
    angles = []
    for i in range(len(vectors)):
        if original_norms[i] > 0 and transformed_norms[i] > 0:
            cos_angle = np.dot(vectors_np[i], transformed[i]) / (original_norms[i] * transformed_norms[i])
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)
        else:
            angles.append(0.0)
    
    return {
        'original_vectors': vectors_np.tolist(),
        'transformed_vectors': transformed.tolist(),
        'original_norms': original_norms.tolist(),
        'transformed_norms': transformed_norms.tolist(),
        'scaling_factors': scaling_factors.tolist(),
        'angles_degrees': angles
    }

# Add these functions to visualizations.py

def compute_matrix_inverse(matrix: List[List[float]]) -> dict:
    """
    Compute the inverse of a matrix with detailed information
    
    Args:
        matrix: n×n matrix
    
    Returns:
        Dictionary with inverse matrix and analysis
    """
    A = validate_matrix(matrix)
    n = A.shape[0]
    
    results = {
        'exists': False,
        'inverse': None,
        'condition_number': np.linalg.cond(A),
        'determinant': np.linalg.det(A),
        'error_message': None
    }
    
    try:
        # Check if matrix is square
        if A.shape[0] != A.shape[1]:
            results['error_message'] = "Matrix must be square to have an inverse"
            return results
        
        # Check determinant
        det = results['determinant']
        if abs(det) < 1e-10:
            results['error_message'] = f"Matrix is singular (determinant = {det:.6e})"
            return results
        
        # Compute inverse
        inv = np.linalg.inv(A)
        
        # Verify inverse by checking A * A⁻¹ ≈ I
        identity = np.eye(n)
        product = A @ inv
        error = np.linalg.norm(product - identity)
        
        results['exists'] = True
        results['inverse'] = inv.tolist()
        results['product_error'] = error
        results['is_exact'] = error < 1e-10
        
        # Compute condition number for numerical stability
        results['is_well_conditioned'] = results['condition_number'] < 1e6
        
    except np.linalg.LinAlgError as e:
        results['error_message'] = f"Linear algebra error: {str(e)}"
    except Exception as e:
        results['error_message'] = f"Error computing inverse: {str(e)}"
    
    return results

def compute_matrix_properties(matrix: List[List[float]]) -> dict:
    """
    Compute comprehensive matrix properties
    
    Args:
        matrix: n×n matrix
    
    Returns:
        Dictionary with matrix properties
    """
    A = validate_matrix(matrix)
    n = A.shape[0]
    
    results = {
        'dimension': n,
        'determinant': np.linalg.det(A),
        'trace': np.trace(A),
        'rank': np.linalg.matrix_rank(A),
        'condition_number': np.linalg.cond(A),
        'frobenius_norm': np.linalg.norm(A, 'fro'),
        'spectral_norm': np.linalg.norm(A, 2),
    }
    
    # Matrix type checks
    results['is_symmetric'] = np.allclose(A, A.T)
    results['is_skew_symmetric'] = np.allclose(A, -A.T)
    results['is_orthogonal'] = np.allclose(A @ A.T, np.eye(n))
    results['is_diagonal'] = np.allclose(A, np.diag(np.diag(A)))
    results['is_upper_triangular'] = np.allclose(A, np.triu(A))
    results['is_lower_triangular'] = np.allclose(A, np.tril(A))
    results['is_identity'] = np.allclose(A, np.eye(n))
    
    # Compute eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        results['eigenvalues'] = eigvals.tolist()
        results['eigenvectors'] = eigvecs.tolist()
        results['has_complex_eigenvalues'] = np.any(np.iscomplex(eigvals))
        results['eigenvalues_real'] = eigvals.real.tolist()
        results['eigenvalues_imag'] = eigvals.imag.tolist()
        
        # Spectral properties
        results['spectral_radius'] = np.max(np.abs(eigvals))
        results['min_eigenvalue_magnitude'] = np.min(np.abs(eigvals))
        results['max_eigenvalue_magnitude'] = np.max(np.abs(eigvals))
        
    except:
        results['eigenvalues'] = None
    
    return results

# ============ TESTING FUNCTIONS ============

if __name__ == "__main__":
    """
    Quick test to verify the functions work correctly
    """
    print("Testing visualization functions...")
    
    # Test 2D transformation
    test_matrix_2d = [[2, 1], [1, 2]]
    test_vectors_2d = [[1, 0], [0, 1], [1, 1]]
    
    print(f"\n1. Testing 2D Transformation with matrix:")
    print(np.array(test_matrix_2d))
    
    fig1 = visualize_2d_transformation(test_matrix_2d, test_vectors_2d)
    print("✓ 2D Transformation function created successfully")
    
    # Test 2D determinant
    print(f"\n2. Testing 2D Determinant with matrix:")
    print(np.array(test_matrix_2d))
    
    fig2, det = visualize_2d_determinant(test_matrix_2d)
    print(f"✓ Determinant calculation: {det}")
    print(f"  Expected: 3.0, Got: {det}")
    assert abs(det - 3.0) < 0.0001, "Determinant calculation incorrect!"
    
    # Test 3D transformation
    test_matrix_3d = [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 2]]
    print(f"\n3. Testing 3D Transformation with matrix:")
    print(np.array(test_matrix_3d))
    
    fig3 = visualize_3d_transformation(test_matrix_3d)
    print("✓ 3D Transformation function created successfully")
    
    # Test eigenvectors
    print(f"\n4. Testing Eigenvector visualization with matrix:")
    print(np.array(test_matrix_2d))
    
    fig4 = visualize_eigen_2d(test_matrix_2d)
    eigvals, _ = np.linalg.eig(test_matrix_2d)
    print(f"✓ Eigenvalues: {eigvals[0]:.2f}, {eigvals[1]:.2f}")
    print("  Expected: 3.0, 1.0")
    
    print("\n✅ All tests passed! Functions are working correctly.")
    print("\nNote: Figures are created but not displayed in test mode.")
    print("Run the Streamlit app to see interactive visualizations.")