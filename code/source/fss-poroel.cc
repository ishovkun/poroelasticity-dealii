
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
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <array>

// custom files
#include <right_hand_side.h>
#include <TensorIndexer.h>
#include <PoroElasticPressureSolver.h>
#include <PoroElasticDisplacementSolver.h>


namespace PoroElasticity {
  // elastic constants
  double E = 7e9;
  double nu = 0.25;
  double biot_coef =0.9;
  double permeability = 1e-9;
  double initial_porosity = 30;
  double viscosity = 1e-3;
  double bulk_density = 2100;
  double fluid_compressibility = 0.00689475729;  // 1e-6 psi
  double time_step = 60*60*24;
  double r_well = 0.5;
  double flow_rate = 1e-2;
  double t_max = time_step*100;

  double lame_constant = E*nu/((1.+nu)*(1.-2.*nu));
  double shear_modulus = 0.5*E/(1+nu);
  double bulk_modulus = lame_constant + 2./3.*shear_modulus;
  double grain_bulk_modulus = bulk_modulus/(1 - bulk_modulus);
  double n_modulus = grain_bulk_modulus/(biot_coef - initial_porosity);
  double m_modulus = (n_modulus/fluid_compressibility) /
    (n_modulus*initial_porosity + 1./fluid_compressibility);

  unsigned int bottom = 0, right = 1, top = 2, left = 3;

  // elasticity BC's
  std::vector<unsigned int> displacement_dirichlet_labels =
    {bottom, top, left, right};
  std::vector<unsigned int> displacement_dirichlet_components =
    {1, 1, 0, 0};
  std::vector<double> displacement_dirichlet_values =
    {0, 0, 0, -0.1};

  std::vector<unsigned int> displacement_neumann_labels = {};
  std::vector<unsigned int> displacement_neumann_components = {};
  std::vector<double>       displacement_neumann_values      = {};

  // pressure BC's
  std::vector<unsigned int> pressure_dirichlet_labels     = {};
  std::vector<unsigned int> pressure_dirichlet_components = {};
  std::vector<double>       pressure_dirichlet_values     = {};

  std::vector<unsigned int> pressure_neumann_labels     = {};
  std::vector<unsigned int> pressure_neumann_components = {};
  std::vector<double>       pressure_neumann_values     = {};

  using namespace dealii;

  // --------------------- Compute local strain ------------------------------
  template <int dim>
  inline SymmetricTensor<2,dim>
  get_local_strain (const std::vector<Tensor<1,dim> > &grad)
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
  // ---------------------------- Problem ------------------------------------
  template <int dim>
  class PoroElasticProblem {
  public:
    PoroElasticProblem();
    ~PoroElasticProblem();
    void run ();

  private:
    void read_mesh();

    void setup_dofs();
    void set_boundary_conditions();

    void assemble_displacement_system_matrix();
    void assemble_displacement_rhs();
    void solve_displacement_system();

    void assemble_strain_projection_matrix();
    void assemble_strain_projection_rhs(std::vector<int> tensor_components);
    void solve_strain_projection(int rhs_entry);
    void get_effective_stresses();
    void get_total_stresses(std::vector<int> tensor_components);

    void get_volumetric_strain();
    void update_volumetric_strain();
    void assemble_pressure_residual();
    void assemble_pressure_jacobian();
    void solve_pressure_system();
    void output_results(const unsigned int time_step_number);

    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    indexing::TensorIndexer<dim>  tensor_indexer;
    Triangulation<dim>            triangulation;
    solvers::PoroElasticPressureSolver<dim>     pressure_solver;
    solvers::PoroElasticDisplacementSolver<dim> displacement_solver;

    // FE_Q<dim>                     pressure_fe;
    // DoFHandler<dim>               pressure_dof_handler;
    // ConstraintMatrix              pressure_constraints;
    // SparsityPattern               pressure_sparsity_pattern;
    // SparseMatrix<double>          pressure_mass_matrix,
    //                               pressure_laplace_matrix,
    //                               pressure_jacobian,
    //                               strain_projection_matrix;

    Vector<double>                volumetric_strain;

    std::vector< Vector<double> > pressure_projection_rhs, strains, stresses;

    // FESystem<dim>                 displacement_fe;
    // DoFHandler<dim>               displacement_dof_handler;
    // ConstraintMatrix              displacement_constraints;
    // SparsityPattern               displacement_sparsity_pattern;
    // SparseMatrix<double>          displacement_system_matrix;
    // Vector<double>                displacement_rhs, displacement_solution;
    // // BoundaryConditions<dim>       displacement_boundary_conditions;
    // SymmetricTensor<4, dim>       get_gassman_tensor(double lambda, double mu);
    // SymmetricTensor<2, dim>       local_strain_tensor(FEValues<dim> &fe_values,
    //                                             const unsigned int shape_func,
    //                                             const unsigned int q_point);
    std::vector<int>              strain_tensor_volumetric_components,
                                  strain_rhs_volumetric_entries,
                                  strain_tensor_shear_components;
    int                           n_stress_components;
  };

  template <int dim>
  PoroElasticProblem<dim>::PoroElasticProblem() :
    pressure_solver(triangulation),
    displacement_solver(triangulation)
  {
    n_stress_components = 0.5*(dim*dim + dim);
    switch (dim) {
    case 1:
      strain_tensor_volumetric_components = {0};
      strain_tensor_shear_components  = {};
      break;
    case 2:
      strain_tensor_volumetric_components = {0, 3};
      strain_tensor_shear_components  = {1};
      break;
    case 3:
      strain_tensor_volumetric_components = {0, 4, 8};
      strain_tensor_shear_components  = {1, 2, 5};
      break;
    default:
      Assert(false, ExcNotImplemented());
    }

    int n_vol_comp = strain_tensor_volumetric_components.size();
    strain_rhs_volumetric_entries.resize(n_vol_comp);
    for(int comp=0; comp<n_vol_comp; ++comp) {
      int strain_rhs_entry =
        tensor_indexer.entryIndex
        (strain_tensor_volumetric_components[comp]);
      strain_rhs_volumetric_entries[comp] = strain_rhs_entry;
    }
  }

  template <int dim>
  PoroElasticProblem<dim>::~PoroElasticProblem()
  {
  }

  // --------------------- Gassman & Strain Tensors --------------------------
  // template <int dim> inline
  // SymmetricTensor<4, dim> PoroElasticProblem<dim>::
  // get_gassman_tensor(double lambda, double mu){
	//   SymmetricTensor<4, dim> tmp;
	//   for (unsigned int i=0; i<dim; ++i)
	// 	  for (unsigned int j=0; j<dim; ++j)
	// 		  for (unsigned int k=0; k<dim; ++k)
	// 			  for (unsigned int l=0; l<dim; ++l)
	// 				  tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
	// 				  	  	  	  	  	 ((i==l) && (j==k) ? mu : 0.0) +
  //                              ((i==j) && (k==l) ? lambda : 0.0));
	//   return tmp;
  // }

  // template <int dim>
  // inline SymmetricTensor<2,dim>
  // PoroElasticProblem<dim>::local_strain_tensor(FEValues<dim> &fe_values,
  //                                              const unsigned int shape_func,
  //                                              const unsigned int q_point) {
	//   SymmetricTensor<2,dim> tmp;
	//   tmp = 0;
	//   for (unsigned int i=0; i<dim; ++i){
	// 	  tmp[i][i] += fe_values.shape_grad_component(shape_func, q_point, i)[i];
  //     for(unsigned int j=0; j<dim;++j){
  //       tmp[i][j] =
  //         (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
  //          fe_values.shape_grad_component(shape_func, q_point, j)[i])/2;
  //     }
	//   }
	//   return tmp;
  // }

  template <int dim>
  void PoroElasticProblem<dim>::setup_dofs()
  {
    pressure_solver.setup_dofs();
    displacement_solver.setup_dofs();
    displacement_solver.set_pressure_fe(pressure_solver.dof_handler,
                                        pressure_solver.fe);

    unsigned int n_dofs = pressure_solver.dof_handler.n_dofs();
    volumetric_strain.reinit(n_dofs);

    //   strain_projection_matrix.reinit(pressure_sparsity_pattern);

    pressure_projection_rhs.resize(n_stress_components);
    strains.resize(n_stress_components);
    stresses.resize(n_stress_components);
    for (int i=0; i<n_stress_components; ++i){
      pressure_projection_rhs[i].reinit(n_dofs);
      strains[i].reinit(n_dofs);
      stresses[i].reinit(n_dofs);
    }

  } // eom

  // template <int dim>
  // void PoroElasticProblem<dim>::
  // assemble_strain_projection_matrix()
  // {
  //   strain_projection_matrix.copy_from(pressure_mass_matrix);
  //   pressure_constraints.condense(strain_projection_matrix);
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::
  // assemble_strain_projection_rhs(std::vector<int> tensor_components)
  // {
  //   // check input
  //   int n_comp = tensor_components.size();

  //   // put assert statements

	//   QGauss<dim>  quadrature_formula(pressure_fe.degree+1);
	//   FEValues<dim> pressure_fe_values(pressure_fe, quadrature_formula,
  //                                    update_values |
  //                                    update_quadrature_points |
  //                                    update_JxW_values);

	//   FEValues<dim> displacement_fe_values(displacement_fe, quadrature_formula,
  //                                        update_gradients |
  //                                        update_quadrature_points);

  //   const unsigned int n_q_points = quadrature_formula.size();
  //   const unsigned int dofs_per_cell = pressure_fe.dofs_per_cell;

  //   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //   // global vectors where to write the RHS values
  //   std::vector<int> entries =
  //     tensor_indexer.entryIndex(tensor_components);

  //   std::vector< Vector<double> > cell_rhs(n_comp,
  //                                          Vector<double>(dofs_per_cell));
  //   for (int c=0; c<n_comp; ++c)
  //     pressure_projection_rhs[entries[c]] = 0;

	//   SymmetricTensor<2, dim> strain_tensor;
  //   std::vector< std::vector<Tensor<1,dim> > >
  //     displacement_grads(quadrature_formula.size(),
  //                        std::vector<Tensor<1,dim> >(dim));

  //   typename DoFHandler<dim>::active_cell_iterator
  //     cell = pressure_dof_handler.begin_active(),
  //     endc = pressure_dof_handler.end(),
  //     displacement_cell = displacement_dof_handler.begin_active();

  //   for (; cell!=endc; ++cell, ++displacement_cell) {
  //     // reinit stuff -------
  //     for (int c=0; c<n_comp; ++c) cell_rhs[c] = 0;
	// 	  pressure_fe_values.reinit(cell);
	// 	  displacement_fe_values.reinit(displacement_cell);
  //     displacement_fe_values.get_function_gradients(displacement_solution,
  //                                                   displacement_grads);
  //     // ----------

  //     for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
  //         strain_tensor = get_local_strain(displacement_grads[q_point]);
  //         double jxw = pressure_fe_values.JxW(q_point);


  //         for (int i=0; i<dofs_per_cell; ++i){
  //           double phi_i = pressure_fe_values.shape_value(i, q_point);

  //           for (int c=0; c<n_comp; ++c) {
  //             int comp = tensor_components[c];
  //             int tensor_component_1 = comp/dim;
  //             int tensor_component_2 = comp%dim;
  //             double strain_value =
  //               strain_tensor[tensor_component_1][tensor_component_2];
  //             // std::cout << strain_value << std::endl;

  //             cell_rhs[c][i] += (phi_i * strain_value * jxw);
  //           }
  //         }

  //     }

  //     // distribute local values to global vectors
  //     cell->get_dof_indices(local_dof_indices);
  //     for (int c=0; c<n_comp; ++c)
  //       pressure_constraints.distribute_local_to_global
  //         (cell_rhs[c], local_dof_indices,
  //          pressure_projection_rhs[entries[c]]);

  //   }

  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::assemble_pressure_jacobian()
  // {
  //   // Accumulation term
  //   pressure_jacobian.copy_from(pressure_mass_matrix);
  //   pressure_jacobian *= (1./m_modulus/time_step);

  //   // Diffusive flow term
  //   double factor = permeability/viscosity;
  //   pressure_jacobian.add(factor, pressure_laplace_matrix);
  //   pressure_constraints.condense(pressure_jacobian);
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::assemble_pressure_residual()
  // {
  //   // Coupling terms
  //   pressure_tmp1 = volumetric_strain;
  //   pressure_tmp1 *= (biot_coef/time_step);

  //   // Accumulation term
  //   pressure_tmp2 = pressure_solution;
  //   pressure_tmp2 -= pressure_old_solution;
  //   pressure_tmp2 *= (1./m_modulus/time_step);
  //   pressure_tmp1 += pressure_tmp2;
  //   pressure_mass_matrix.vmult(pressure_residual, pressure_tmp1);

  //   // Diffusive flow term
  //   pressure_laplace_matrix.vmult(pressure_tmp1, pressure_solution);
  //   double factor = permeability/viscosity;
  //   pressure_tmp1 *= factor;
  //   pressure_residual += pressure_tmp1;

  //   // Source term
  //   right_hand_side::SinglePhaseWell<dim> pressure_source_term_function(r_well);
  //   VectorTools::create_right_hand_side(pressure_dof_handler,
  //                                       QGauss<dim>(pressure_fe.degree+1),
  //                                       pressure_source_term_function,
  //                                       pressure_tmp1);
  //   pressure_residual += pressure_tmp1;

  //   // we are solving jacobian*dp = -residual
  //   pressure_residual *= -1;
  //   pressure_constraints.condense(pressure_residual);
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::solve_strain_projection(int rhs_entry)
  // {
  //   // select entries
  //   auto& rhs_vector = pressure_projection_rhs[rhs_entry];
  //   auto& solution_vector = strains[rhs_entry];

  //   // solve
  //   SolverControl solver_control(1000, 1e-8 * rhs_vector.l2_norm());
  //   SolverCG<> cg(solver_control);
  //   PreconditionSSOR<> preconditioner;
  //   preconditioner.initialize(strain_projection_matrix, 1.0);
  //   cg.solve(strain_projection_matrix, solution_vector, rhs_vector,
  //            preconditioner);
  //   pressure_constraints.distribute(solution_vector);
  //   // std::cout << "     "
  //   //           << "Projection component: "
  //   //           << rhs_entry
  //   //           << " CG iterations: "
  //   //           << solver_control.last_step()
  //   //           << std::endl;
  //   // std::cout << "     "
  //   //           << "RHS norm "
  //   //           << rhs_vector.l2_norm()
  //   //           << " solution: "
  //   //           << solution_vector.l2_norm()
  //   //           << " solution1: "
  //   //           << strains[rhs_entry].l2_norm()
  //   //           << std::endl
  //   //           << std::endl;
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::solve_displacement_system()
  // {
  //   // double solver_tolerance = 1e-8*displacement_rhs.l2_norm();
  //   double solver_tolerance = 1e-12;
  //   SolverControl           solver_control(1000, solver_tolerance);
  //   SolverCG<>              cg (solver_control);

  //   PreconditionSSOR<> preconditioner;
  //   preconditioner.initialize(displacement_system_matrix, 1.2);

  //   // displacement_solution = 0;
  //   cg.solve(displacement_system_matrix,
  //            displacement_solution, displacement_rhs,
  //            preconditioner);

  //   displacement_constraints.distribute(displacement_solution);
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::get_volumetric_strain()
  // {
  //   volumetric_strain = 0;
  //   for(const auto &comp : strain_rhs_volumetric_entries) {
  //     volumetric_strain += strains[comp];
  //   }
  // }

  template <int dim>
  void PoroElasticProblem<dim>::update_volumetric_strain()
  {
    pressure_solver.tmp1 = pressure_solver.solution_update;
    pressure_solver.tmp1 *= (biot_coef/bulk_modulus);
    volumetric_strain += pressure_solver.tmp1;
  }

  // template <int dim>
  // void PoroElasticProblem<dim>::
  // get_effective_stresses()
  // {
  //   SymmetricTensor<2,dim> node_strain_tensor, node_stress_tensor;
  //   SymmetricTensor<4,dim> gassman_tensor =
  //     get_gassman_tensor(lame_constant, shear_modulus);

  //   // iterate over nodes
  //   unsigned int pressure_n_dofs = pressure_dof_handler.n_dofs();
  //   for (unsigned int l=0; l<pressure_n_dofs; ++l)
  //     {
  //       // iterate over components within each node
  //       for (int i=0; i<dim; ++i){
  //         // note that j is from i to dim since the tensor is symmetric
  //         for (int j=i; j<dim; ++j){
  //           int strain_entry =
  //             tensor_indexer.entryIndex(i*dim + j);
  //           double strain_value = strains[strain_entry][l];
  //           node_strain_tensor[i][j] = strain_value;
  //           // since it's symmetric
  //           if (i != j) node_strain_tensor[j][i] = strain_value;
  //         }
  //       }
  //       node_stress_tensor = gassman_tensor*node_strain_tensor;
  //       // distribute node stress tensor values to global vectors
  //       for (int i=0; i<dim; ++i){
  //         for (int j=i; j<dim; ++j){
  //           int stress_component =
  //             tensor_indexer.entryIndex(i*dim + j);
  //           stresses[stress_component][l] = node_stress_tensor[i][j];
  //         }
  //       }
  //   }
  // }

  // template <int dim>
  // void PoroElasticProblem<dim>::output_results(const unsigned int time_step_number)
  // {
  //   std::vector<std::string> displacement_names(dim, "u");

  //   std::vector<DataComponentInterpretation::DataComponentInterpretation>
  //   displacement_component_interpretation
  //     (dim, DataComponentInterpretation::component_is_part_of_vector);

  //   DataOut<dim> data_out;
  //   data_out.add_data_vector(displacement_dof_handler, displacement_solution,
  //                            displacement_names,
  //                            displacement_component_interpretation);

  //   data_out.add_data_vector(pressure_dof_handler, pressure_solution, "p");

  //   data_out.add_data_vector(pressure_dof_handler, strains[0], "eps_xx");
  //   data_out.add_data_vector(pressure_dof_handler, stresses[0], "sigma_xx");

  //   switch (dim) {
  //   case 2:
  //     data_out.add_data_vector(pressure_dof_handler, strains[1], "eps_xy");
  //     data_out.add_data_vector(pressure_dof_handler, strains[2], "eps_yy");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[1], "sigma_xy");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[2], "sigma_yy");
  //     break;
  //   case 3:
  //     data_out.add_data_vector(pressure_dof_handler, stresses[1], "sigma_xy");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[2], "sigma_xz");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[3], "sigma_yy");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[4], "sigma_yz");
  //     data_out.add_data_vector(pressure_dof_handler, stresses[5], "sigma_zz");
  //     break;
  //   }

  //   data_out.build_patches(std::min(displacement_fe.degree,
  //                                   pressure_fe.degree));
  //   std::ostringstream filename;
  //   filename << "./solution/solution-" <<
  //     Utilities::int_to_string(time_step_number, 4) << ".vtk";
  //   std::ofstream output (filename.str().c_str());
  //   data_out.write_vtk(output);
  // }


  template <int dim>
  void PoroElasticProblem<dim>::run()
  {
    const unsigned int initial_global_refinement = 1;
    const unsigned int n_adaptive_pre_refinement_steps = 3;
    read_mesh();

    triangulation.refine_global(initial_global_refinement);

    displacement_solver.set_boundary_conditions(
        displacement_neumann_labels,
        displacement_neumann_components,
        displacement_neumann_values,
        displacement_dirichlet_labels,
        displacement_dirichlet_components,
        displacement_dirichlet_values);

    setup_dofs();

    // initial domain variables
    pressure_solver.solution = 0;
    volumetric_strain = 0;
    displacement_solver.assemble_system(pressure_solver.solution);
    // assemble_strain_projection_matrix();

    double time = 0;
    unsigned int time_step_number = 0;
    double fss_TOL = 1e-8;
    double pressure_TOL = 1e-8;
    double pressure_error;
    int max_pressure_iterations = 100;
    int max_fss_iterations = 100;

    // while (time < t_max){
    // // while (time < time_step*20){
    while (time < time_step){
      time += time_step;
      time_step_number++;
      std::cout << "Time: " << time << std::endl;
      // std::cout << "   av pressure: " << pressure_solution.l2_norm() << std::endl;

      // if (time_step_number % 5 == 0){
      //   std::cout << "Refining mesh" << std::endl;
      //   refine_mesh(initial_global_refinement,
      //               initial_global_refinement +
      //               n_adaptive_pre_refinement_steps);
      //   assemble_displacement_system_matrix();
      //   assemble_strain_projection_matrix();
      // }

      pressure_solver.old_solution = pressure_solver.solution;

      // // remove this section (for development purposes)
      // pressure_solver.assemble_residual(time_step, volumetric_strain);
      // update_volumetric_strain();
      // pressure_solver.assemble_jacobian(time_step);
      // pressure_solver.solve();
      // displacement_solver.solve();
      // // EOS

      // strart Fixed-stress-split iterations
      pressure_error = pressure_TOL*2;
      int fss_iteration = 0;
      while (fss_iteration < max_fss_iterations && pressure_error > pressure_TOL)
        {
          fss_iteration++;
          std::cout << "    Coupling iteration: " << fss_iteration << std::endl;

          // pressure iterations
          pressure_solver.solution_update = 0;
          int pressure_iteration = 0;
          while (pressure_iteration < max_pressure_iterations) {
            pressure_iteration++;
            update_volumetric_strain();
            pressure_solver.assemble_residual(time_step, volumetric_strain);
            pressure_error = pressure_solver.residual.l2_norm();
            if (pressure_error < pressure_TOL) {
              std::cout << "        pressure converged; iterations: "
                        << pressure_iteration
                        << std::endl;
              break;
            }

            std::cout << "     "
                      << "Pressure iteration: " << pressure_iteration
                      << "; error: " << pressure_error << std::endl;
            pressure_solver.assemble_jacobian(time_step);
            pressure_solver.solve();
          } // end pressure iterations

          // Solve displacement system
          displacement_solver.assemble_system(pressure_solver.solution);
          displacement_solver.solve();

          // // compute components of volumetric strains
          // assemble_strain_projection_rhs(strain_tensor_volumetric_components);
          // for(const auto &comp : strain_tensor_volumetric_components ) {
          //   int strain_rhs_entry =
          //     tensor_indexer.entryIndex(comp);
          //   solve_strain_projection(strain_rhs_entry);
          //   // std::cout << strain_rhs_entry << std::endl;
          //   // std::cout << comp << std::endl;
          // }

          // get_volumetric_strain();

          // get error
          pressure_solver.assemble_residual(time_step, volumetric_strain);
          pressure_error = pressure_solver.residual.l2_norm();
          std::cout << "        Error: " << pressure_error << std::endl;
        } // end FSS iterations

      // compute shear strains only since volumetric are alredy obtained
      // for(const auto &comp : strain_tensor_shear_components) {
      //   int strain_rhs_entry =
      //     tensor_indexer.entryIndex(comp);
      //   solve_strain_projection(strain_rhs_entry);
      // }

      // get_effective_stresses();
      // output_results(time_step_number);

    }  // end time stepping

  }  // EOM

  template <int dim>
  void PoroElasticProblem<dim>::read_mesh ()
  {
	  GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f("domain.msh");
	  gridin.read_msh(f);
  }

  // template <int dim>
  // void PoroElasticProblem<dim>::refine_mesh(const unsigned int min_grid_level,
  //                                           const unsigned int max_grid_level)

  // {
  //   Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  //   int fe_degree = pressure_fe.degree;
  //   KellyErrorEstimator<dim>::estimate (pressure_dof_handler,
  //                                       QGauss<dim-1>(fe_degree+1),
  //                                       typename FunctionMap<dim>::type(),
  //                                       pressure_solution,
  //                                       estimated_error_per_cell);

  //   // int fe_degree = displacement_fe.degree;
  //   // KellyErrorEstimator<dim>::estimate (displacement_dof_handler,
  //   //                                     QGauss<dim-1>(fe_degree+1),
  //   //                                     typename FunctionMap<dim>::type(),
  //   //                                     displacement_solution,
  //   //                                     estimated_error_per_cell);
  //   GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
  //                                                     estimated_error_per_cell,
  //                                                     0.6, 0.4);
  //   if (triangulation.n_levels() > max_grid_level)
  //     for (typename Triangulation<dim>::active_cell_iterator
  //            cell = triangulation.begin_active(max_grid_level);
  //          cell != triangulation.end(); ++cell)
  //       cell->clear_refine_flag();
  //   for (typename Triangulation<dim>::active_cell_iterator
  //          cell = triangulation.begin_active(min_grid_level);
  //        cell != triangulation.end_active(min_grid_level); ++cell)
  //     cell->clear_coarsen_flag();

  //   // Pressure solution transfer
  //   SolutionTransfer<dim> pressure_trans(pressure_dof_handler);
  //   std::vector< Vector<double> > previous_solution(2);
  //   previous_solution[0] = pressure_solution;
  //   previous_solution[1] = volumetric_strain;

  //   triangulation.prepare_coarsening_and_refinement();
  //   pressure_trans.prepare_for_coarsening_and_refinement(previous_solution);
  //   triangulation.execute_coarsening_and_refinement();

  //   setup_dofs();

  //   std::vector< Vector<double> > tmp(2);
  //   tmp[0].reinit(pressure_solution);
  //   tmp[1].reinit(volumetric_strain);
  //   pressure_trans.interpolate(previous_solution, tmp);

  //   pressure_solution = tmp[0];
  //   volumetric_strain = tmp[1];
  //   // ???
  //   // pressure_constraints.distribute(pressure_solution);
  //   // pressure_constraints.distribute(volumetric_strain);
  // }

  // end of namespace
}


int main () {
  try {
    dealii::deallog.depth_console(0);

    PoroElasticity::PoroElasticProblem<2> poro_elastic_problem_2d;
    poro_elastic_problem_2d.run();
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
