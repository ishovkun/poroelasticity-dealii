/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


// @sect3{Include files}

// As usual, the first few include files are already known, so we will not
// comment on them further.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/tensor_function.h>
// In this example, we need vector-valued finite elements. The support for
// these can be found in the following include file:
#include <deal.II/fe/fe_system.h>
// We will compose the vector-valued finite elements from regular Q1 elements
// which can be found here, as usual:
#include <deal.II/fe/fe_q.h>

// This again is C++:
#include <fstream>
#include <iostream>
#include <vector>
#include <array>



namespace Step8
{
  using namespace dealii;

  double E = 1e6;
  double nu = 0.25;
  double lambda_constant_value = E*nu/((1.+nu)*(1.-2.*nu));
  double mu_constant_value = 0.5*E/(1+nu);

  ConstantFunction<2> lambda(lambda_constant_value), mu(mu_constant_value);
  
  template <int dim>
  void print_mesh_info(const Triangulation<dim> &tria,
                       const std::string        &filename)
  {
    std::cout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << tria.n_active_cells() << std::endl;
    {
      std::map<unsigned int, unsigned int> boundary_count;
      typename Triangulation<dim>::active_cell_iterator
        cell = tria.begin_active(),
        endc = tria.end();
      for (; cell!=endc; ++cell)
        {
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary())
                boundary_count[cell->face(face)->boundary_id()]++;
            }
        }
      std::cout << " boundary indicators: ";
      for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
           it!=boundary_count.end();
           ++it)
        {
          std::cout << it->first << "(" << it->second << " times) ";
        }
      std::cout << std::endl;
    }
    std::ofstream out (filename.c_str());
    GridOut grid_out;
    grid_out.write_eps (tria, out);
    std::cout << " written to " << filename
              << std::endl
              << std::endl;
  }

  
  template <int dim>
  class RightHandSide :  public Function<dim>
  {
  public:
    RightHandSide ();

    virtual void vector_value (Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };


  template <int dim>
  RightHandSide<dim>::RightHandSide ()
    :
    Function<dim> (dim)
  {}

  template <int dim>
  inline
  void RightHandSide<dim>::vector_value (Vector<double>   &values) const {
    Assert(values.size() == dim,
           ExcDimensionMismatch (values.size(), dim));
    Assert(dim >= 2, ExcNotImplemented());

    values(0) = 0;
    values(1) = 0;
  }

  template <int dim>
  void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> >   &value_list
                                             ) const {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value (value_list[p]);
  }

  template <int dim>
  inline
  SymmetricTensor<2,dim>
  get_strain (const std::vector<Tensor<1,dim> > &grad)
  {
    Assert (grad.size() == dim, ExcInternalError());
    SymmetricTensor<2,dim> strain;
    for (unsigned int i=0; i<dim; ++i)
      strain[i][i] = grad[i][i];
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=i+1; j<dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
    return strain;
  }

  template <int dim>
  class ElasticProblem {
  public:
    ElasticProblem ();
    ~ElasticProblem ();
    void run ();

  private:
    void read_mesh();
    void setup_system ();
    void assemble_system ();
    // void assemble_strain_system ();

    SymmetricTensor<2,dim> local_strain_tensor(FEValues<dim> &fe_values,
    										   const unsigned int shape_func,
											   const unsigned int q_point);

    SymmetricTensor<4,dim> get_gassman_tensor(double lambda,double mu);

    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;
    void compute_derived_quantities();

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    ConstraintMatrix constraints;
    SparsityPattern sparsity_pattern;

    SparseMatrix<double> system_matrix;

    Vector<double> displacement, system_rhs;
    
    Vector<double> sigma_xx, sigma_yy, sigma_zz,
      sigma_xy, sigma_xz, sigma_yz;

    std::vector<unsigned int> dirichlet_boundary_labels,
      neumann_boundary_labels, dirichlet_components,
      neumann_components;

    std::vector<double> dirichlet_boundary_values,
      neumann_boundary_values;
  };

  
  template <int dim>
  ElasticProblem<dim>::ElasticProblem():
    dof_handler(triangulation),
    fe(FE_Q<dim>(1), dim)
  {}


  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }

  
  template <int dim>
  void ElasticProblem<dim>::read_mesh (){
	  GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f("domain.msh");
	  gridin.read_msh(f);
    print_mesh_info(triangulation, "g1.eps");
    
    unsigned int
      bottom = 0,
      right = 1,
      top = 2,
      left = 3,
      wellbore = 4;
    
    // dirichlet_boundary_labels = {wellbore, wellbore};
	  // dirichlet_components = 	  {0, 1};
	  // dirichlet_boundary_values = {0, 0};

    double sx = -5e6, sy = -5e6, p_well = 2e6;
    
    dirichlet_boundary_labels = {bottom, left};
	  dirichlet_components = 	{1, 0};
	  dirichlet_boundary_values = {0, 0};

	  neumann_boundary_labels = {wellbore, wellbore, right, top};
	  neumann_boundary_values = {-p_well, -p_well, sx, sy};
	  neumann_components =		{0, 1, 0, 1};


  }

 
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints);

    std::vector<ComponentMask> displacement_masks(dim);
    for (unsigned int comp=0; comp<dim; ++comp){
    	FEValuesExtractors::Scalar displacement_extractor(comp);
    	displacement_masks[comp] = fe.component_mask(displacement_extractor);
    }

    unsigned int n_dirichlet_conditions = dirichlet_boundary_labels.size();
    
    for (unsigned int cond=0;cond<n_dirichlet_conditions;++cond){
    	unsigned int component = dirichlet_components[cond];
    	double dirichlet_value = dirichlet_boundary_values[cond];
    	VectorTools::interpolate_boundary_values(dof_handler,
				 dirichlet_boundary_labels[cond],
				 ConstantFunction<dim>(dirichlet_value, dim),
				 constraints,
				 displacement_masks[component]);
    }
    constraints.close ();

    unsigned int n_dofs = dof_handler.n_dofs();
    unsigned int n_cells = triangulation.n_active_cells();
    DynamicSparsityPattern dsp(n_dofs, n_dofs);
    
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp, constraints,
                                    /*keep_constrained_dofs = */ false);
    
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    displacement.reinit(n_dofs);
    system_rhs.reinit(n_dofs);
    switch (dim) {
    case 1:
      sigma_xx.reinit(n_cells);
    case 2:
      sigma_xx.reinit(n_cells);
      sigma_xy.reinit(n_cells);
      sigma_yy.reinit(n_cells);
    case 3:
      sigma_xx.reinit(n_cells);
      sigma_xy.reinit(n_cells);
      sigma_yy.reinit(n_cells);
      sigma_zz.reinit(n_cells);
      sigma_yz.reinit(n_cells);
      sigma_xz.reinit(n_cells);
    }
  }


  template <int dim>
  inline SymmetricTensor<2,dim>
  ElasticProblem<dim>::local_strain_tensor(FEValues<dim> &fe_values,
                                           const unsigned int shape_func,
                                           const unsigned int q_point) {
	  SymmetricTensor<2,dim> tmp;
	  tmp = 0;
	  for (unsigned int i=0; i<dim; ++i){
		  tmp[i][i] += fe_values.shape_grad_component(shape_func, q_point, i)[i];
	  	  for(unsigned int j=0; j<dim;++j){
	  		  tmp[i][j] =
            (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
             fe_values.shape_grad_component(shape_func,q_point,j)[i])/2;
	  	  }
	  }
	  return tmp;
  }

  template <int dim> inline
  SymmetricTensor<4,dim> ElasticProblem<dim>::
  get_gassman_tensor(double lambda, double mu){
	  SymmetricTensor<4,dim> tmp;
	  for (unsigned int i=0;i<dim;++i)
		  for (unsigned int j=0;j<dim;++j)
			  for (unsigned int k=0;k<dim;++k)
				  for (unsigned int l=0;l<dim;++l)
					  tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
					  	  	  	  	  	 ((i==l) && (j==k) ? mu : 0.0) +
                               ((i==j) && (k==l) ? lambda : 0.0));
	  return tmp;
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system ()
  {
    QGauss<dim>  quadrature_formula(2);
    QGauss<dim-1>  face_quadrature_formula(2);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,
                                     update_values |
                                     update_quadrature_points |
                                     update_normal_vectors |
                                     update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);
    RightHandSide<dim>  right_hand_side;
    std::vector< Vector<double> > rhs_values(n_q_points,
                                             Vector<double>(dim));

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    
    SymmetricTensor<2,dim>	strain_tensor_i;
    SymmetricTensor<2,dim>	strain_tensor_j;
    SymmetricTensor<4,dim>	gassman_tensor;
    Tensor<1,dim>	neumann_bc_vector;

    for (; cell!=endc; ++cell){
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);

      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                        rhs_values);

      for (unsigned int i=0; i<dofs_per_cell; ++i){
        for (unsigned int j=0; j<dofs_per_cell; ++j){
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
            gassman_tensor = get_gassman_tensor(lambda_values[q_point],
                                                mu_values[q_point]);
            strain_tensor_i = local_strain_tensor(fe_values,i,q_point);
            strain_tensor_j = local_strain_tensor(fe_values,j,q_point);
            cell_matrix(i,j) +=
              gassman_tensor*strain_tensor_i*strain_tensor_j*fe_values.JxW(q_point);
          }
        }
      }


      for (unsigned int i=0; i<dofs_per_cell; ++i){
        const unsigned int
          component_i = fe.system_to_component_index(i).first;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          cell_rhs(i) += fe_values.shape_value(i,q_point) *
            rhs_values[q_point](component_i) *
            fe_values.JxW(q_point);
      }
      
      // impose neumann conditions
      // iterate through faces
      for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f){
        if (cell->face(f)->at_boundary()) {
          unsigned int n_neumann_conditions = neumann_boundary_labels.size();
          unsigned int face_boundary_id = cell->face(f)->boundary_id();
          fe_face_values.reinit(cell, f);
          
          // loop through different boundary labels
          for (unsigned int l=0; l < n_neumann_conditions; ++l){
            int id = neumann_boundary_labels[l];
              
            if (face_boundary_id == id) {
              for (unsigned int i=0; i<dofs_per_cell; ++i){
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;

                if (component_i == neumann_components[l]) {
                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) {
                    double neumann_value = neumann_boundary_values[l] *
                      fe_face_values.normal_vector(q_point)[component_i];
                  
                    cell_rhs(i) +=
                      fe_face_values.shape_value(i, q_point) *
                      neumann_value *
                      fe_face_values.JxW(q_point);
                  }
                }
              }
            }
          }
        }
      }

      // impose Dirichlet conditions
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  }


  template <int dim>
  void ElasticProblem<dim>::compute_derived_quantities(){
	  QGauss<dim>  quadrature_formula(2);
	  FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

	  const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    // const unsigned int n_dofs = dof_handler.n_dofs();

    std::vector<types::global_dof_index>
      local_dof_indices (dofs_per_cell);

    SymmetricTensor<4,dim> gassman_tensor =
      get_gassman_tensor(lambda_constant_value, mu_constant_value);

	  SymmetricTensor<2,dim> strain_tensor, temp;
	  std::vector< SymmetricTensor<2,dim> > node_strains(n_q_points);

    std::vector< std::vector<Tensor<1,dim> > >
      displacement_grads(quadrature_formula.size(),
                         std::vector<Tensor<1,dim> >(dim));


    SymmetricTensor<2,dim> local_stress;
    
	  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    

    unsigned cell_index = 0;
	  for (; cell!=endc; ++cell) {
		  fe_values.reinit(cell);
      fe_values.get_function_gradients(displacement, displacement_grads);
      local_stress.clear();
 
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
        local_stress +=
            gassman_tensor *
            get_strain(displacement_grads[q_point]);
      }
      local_stress /= n_q_points;
      
      switch (dim) {
      case 1:
        sigma_xx[cell_index] = local_stress[0][0];
        break;
      case 2:
        sigma_xx[cell_index] = local_stress[0][0];
        sigma_xy[cell_index] = local_stress[0][1];
        sigma_yy[cell_index] = local_stress[1][1];
        break;
      case 3:
        sigma_xx[cell_index] = local_stress[0][0];
        sigma_xy[cell_index] = local_stress[0][1];
        sigma_xz[cell_index] = local_stress[0][2];
        sigma_yy[cell_index] = local_stress[1][1];
        sigma_yz[cell_index] = local_stress[1][2];
        sigma_zz[cell_index] = local_stress[2][2];
        break;
      }
      // std::cout << "stress: " << local_stress[0][0] << std::endl;
      cell_index ++;
    }
  }

  template <int dim>
  void ElasticProblem<dim>::solve() {
    SolverControl           solver_control (1000, 1e-12); // maxiter,presicion
    SolverCG<>              cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, displacement, system_rhs,
              preconditioner);
   
    constraints.distribute (displacement);
  }


  template <int dim>
  void ElasticProblem<dim>::refine_grid() {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(2),
                                        typename FunctionMap<dim>::type(),
                                        displacement,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.03);

    triangulation.execute_coarsening_and_refinement ();
  }


  template <int dim>
  void ElasticProblem<dim>::output_results (const unsigned int cycle) const
  {
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".vtk";
    std::ofstream output (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);

    // write displacements
    std::vector<std::string> names1;
    switch (dim) {
      case 1:
        names1.push_back("displacement");
        data_out.add_data_vector(sigma_xx, "sigma_xx");
        break;
      case 2:
        names1.push_back("x_displacement");
        names1.push_back("y_displacement");
        break;
      case 3:
        names1.push_back("x_displacement");
        names1.push_back("y_displacement");
        names1.push_back("z_displacement");
        break;
      default:
        Assert(false, ExcNotImplemented());
      }

    data_out.add_data_vector(displacement, names1);

    // write stresses
    // std::vector<std::string> names2;
    switch (dim) {
    case 1:
      data_out.add_data_vector(sigma_xx, "sigma_xx");
      break;
    case 2:
      data_out.add_data_vector(sigma_xx, "sigma_xx");
      data_out.add_data_vector(sigma_xy, "sigma_xy");
      data_out.add_data_vector(sigma_yy, "sigma_yy");
      break;
    case 3:
      data_out.add_data_vector(sigma_xx, "sigma_xx");
      data_out.add_data_vector(sigma_xy, "sigma_xy");
      data_out.add_data_vector(sigma_xz, "sigma_xz");
      data_out.add_data_vector(sigma_yy, "sigma_yy");
      data_out.add_data_vector(sigma_yz, "sigma_yz");
      data_out.add_data_vector(sigma_zz, "sigma_zz");
    }

    data_out.build_patches();
    data_out.write_vtk(output);
  }


  template <int dim>
  void ElasticProblem<dim>::run () {
    for (unsigned int cycle=0; cycle<3; ++cycle) {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        read_mesh();
      else
        refine_grid();

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      
      compute_derived_quantities();
      output_results (cycle);
    }
  }
  
}

int main () {
  try {
    dealii::deallog.depth_console (0);
    
    Step8::ElasticProblem<2> elastic_problem_2d;
    elastic_problem_2d.run ();
  }
  
  catch (std::exception &exc) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  
  catch(...) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
 
  return 0;
}

