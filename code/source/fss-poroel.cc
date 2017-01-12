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

// custom modules
#include <right_hand_side.h>
#include <TensorIndexer.h>
#include <PoroElasticPressureSolver.h>
#include <PoroElasticDisplacementSolver.h>
#include <StrainProjector.h>
// #include <ConstitutiveModel.h>


namespace PoroElasticity {

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

    void solve_strain_projection(int rhs_entry);
    void get_effective_stresses();
    void get_total_stresses(std::vector<int> tensor_components);

    void get_normal_strain_components();
    void get_shear_strain_components();
    void get_volumetric_strain();
    void output_results(const unsigned int time_step_number);
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    // variables
    Triangulation<dim>                          triangulation;
    input_data::InputDataPoroel<dim>            data;
    solvers::PoroElasticPressureSolver<dim>     pressure_solver;
    solvers::PoroElasticDisplacementSolver<dim> displacement_solver;
    projection::StrainProjector<dim>            strain_projector;

    indexing::TensorIndexer<dim>  tensor_indexer;

    Vector<double>                volumetric_strain, initial_volumetric_strain;
    std::vector< Vector<double> > strains, stresses;

    std::vector<int>              strain_tensor_volumetric_components,
                                  strain_rhs_volumetric_entries,
                                  strain_tensor_shear_components;
    int                           n_stress_components;
  };

  template <int dim>
  PoroElasticProblem<dim>::PoroElasticProblem() :
    pressure_solver(triangulation, data),
    displacement_solver(triangulation, data)
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
  {}


  template <int dim>
  void PoroElasticProblem<dim>::setup_dofs()
  {
    pressure_solver.setup_dofs();
    displacement_solver.setup_dofs();

    // let solvers know about each other's fe and dofs
    displacement_solver.set_pressure_fe(pressure_solver.dof_handler,
                                        pressure_solver.fe);
    strain_projector.set_solvers(displacement_solver, pressure_solver);

    strain_projector.setup_dofs();

    unsigned int n_dofs = pressure_solver.dof_handler.n_dofs();
    volumetric_strain.reinit(n_dofs);
    initial_volumetric_strain.reinit(n_dofs);

    stresses.resize(n_stress_components);
    for (int i=0; i<n_stress_components; ++i)
      stresses[i].reinit(n_dofs);
  } // eom

  template <int dim>
  void PoroElasticProblem<dim>::get_normal_strain_components()
  {
    // compute components of volumetric strains
    strain_projector.assemble_projection_rhs(strain_tensor_volumetric_components);
    for(const auto &comp : strain_tensor_volumetric_components ) {
      int strain_rhs_entry = tensor_indexer.entryIndex(comp);
      strain_projector.solve_projection_system(strain_rhs_entry);
      // std::cout << strain_rhs_entry << std::endl;
      // std::cout << comp << std::endl;
    }
  }  // EOM


  template <int dim>
  void PoroElasticProblem<dim>::get_shear_strain_components()
  {
    // compute shear strains only since volumetric are alredy obtained
    for(const auto &comp : strain_tensor_shear_components) {
      int strain_rhs_entry =
        tensor_indexer.entryIndex(comp);
      strain_projector.solve_projection_system(strain_rhs_entry);
    }
  }  // EOM


  template <int dim>
  void PoroElasticProblem<dim>::get_volumetric_strain()
  {
    // compute volumetric strain
    volumetric_strain = 0;
    for(const auto &comp : strain_rhs_volumetric_entries)
      volumetric_strain += strain_projector.strains[comp];
  }  // EOM


  template <int dim>
  void PoroElasticProblem<dim>::
  get_effective_stresses()
  {
    SymmetricTensor<2,dim> node_strain_tensor, node_stress_tensor;
    SymmetricTensor<4,dim> gassman_tensor =
      constitutive_model::isotropic_gassman_tensor<dim>(data.lame_constant,
                                                        data.shear_modulus);

    // iterate over nodes
    unsigned int pressure_n_dofs = pressure_solver.dof_handler.n_dofs();
    for (unsigned int l=0; l<pressure_n_dofs; ++l)
      {
        // iterate over components within each node
        for (int i=0; i<dim; ++i){
          // note that j is from i to dim since the tensor is symmetric
          for (int j=i; j<dim; ++j){
            int strain_entry =
              tensor_indexer.entryIndex(i*dim + j);
            double strain_value = strain_projector.strains[strain_entry][l];
            node_strain_tensor[i][j] = strain_value;
            // since it's symmetric
            if (i != j) node_strain_tensor[j][i] = strain_value;
          }
        }
        node_stress_tensor = gassman_tensor*node_strain_tensor;
        // distribute node stress tensor values to global vectors
        for (int i=0; i<dim; ++i){
          for (int j=i; j<dim; ++j){
            int stress_component =
              tensor_indexer.entryIndex(i*dim + j);
            stresses[stress_component][l] = node_stress_tensor[i][j];
          }
        }
      }  // end loop over nodes
  }  // EOM


  template <int dim>
  void PoroElasticProblem<dim>::output_results(const unsigned int time_step_number)
  {
    std::vector<std::string> displacement_names(dim, "u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    displacement_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.add_data_vector(displacement_solver.dof_handler,
                             displacement_solver.solution,
                             displacement_names,
                             displacement_component_interpretation);

    data_out.add_data_vector(pressure_solver.dof_handler,
                             pressure_solver.solution, "p");

    data_out.add_data_vector(pressure_solver.dof_handler,
                             strain_projector.strains[0], "eps_xx");
    data_out.add_data_vector(pressure_solver.dof_handler,
                             stresses[0], "sigma_xx");
    switch (dim) {
    case 2:
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[1], "eps_xy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[2], "eps_yy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[1], "sigma_xy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[0], "sigma_yy");
      break;
    case 3:
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[1], "eps_xy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[2], "eps_xz");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[3], "eps_yy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[4], "eps_yz");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               strain_projector.strains[5], "eps_zz");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[1], "sigma_xy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[2], "sigma_xz");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[3], "sigma_yy");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[4], "sigma_yz");
      data_out.add_data_vector(pressure_solver.dof_handler,
                               stresses[5], "sigma_zz");
      break;
    }

    data_out.build_patches(std::min(displacement_solver.fe.degree,
                                    pressure_solver.fe.degree));
    std::ostringstream filename;
    filename << "./solution/solution-" <<
      Utilities::int_to_string(time_step_number, 4) << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk(output);
  }


  template <int dim>
  void PoroElasticProblem<dim>::run()
  {
    data.read_input_file("input.data");
    read_mesh();

    const unsigned int initial_global_refinement = 1;
    const unsigned int n_adaptive_pre_refinement_steps = 3;
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
    // volumetric_strain = 0;
    displacement_solver.assemble_system(pressure_solver.solution);

    // compute initial state
    displacement_solver.solve();
    strain_projector.assemble_projection_matrix();
    get_normal_strain_components();
    get_volumetric_strain();
    initial_volumetric_strain = volumetric_strain;

    double time = 0;
    double time_step = data.time_step;
    unsigned int time_step_number = 0;
    double pressure_error;

    while (time < data.t_max){
      time += time_step;
      time_step_number++;
      std::cout << "Time: " << time << std::endl;
      // std::cout << "   av pressure: " << pressure_solution.l2_norm() << std::endl;

      if (time_step_number % 5 == 0){
        std::cout << "Refining mesh" << std::endl;
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                    n_adaptive_pre_refinement_steps);
        displacement_solver.assemble_system(pressure_solver.solution);
        strain_projector.assemble_projection_matrix();
      }

      pressure_solver.old_solution = pressure_solver.solution;

      // strart Fixed-stress-split iterations
      pressure_error = data.pressure_tol*2;
      int fss_iteration = 0;
      while (fss_iteration < data.max_fss_iterations &&
             pressure_error > data.fss_tol)
      // while (fss_iteration < 1)
        {
          fss_iteration++;
          std::cout << "    Coupling iteration: " << fss_iteration << std::endl;

          // pressure iterations
          int pressure_iteration = 0;
          pressure_solver.solution_update = 0;

          while (pressure_iteration < data.max_pressure_iterations) {
            pressure_iteration++;
            // pressure_solver.update_volumetric_strain(volumetric_strain);
            pressure_solver.assemble_residual(time_step,
                                              volumetric_strain,
                                              initial_volumetric_strain);
            pressure_error = pressure_solver.residual.l2_norm();

            if (pressure_error < data.pressure_tol) {
              std::cout << "        pressure converged; iterations: "
                        << pressure_iteration - 1
                        << std::endl;
              break;
            }

            // std::cout << "     "
            //           << "Pressure iteration: " << pressure_iteration
            //           << "; error: " << pressure_error << std::endl;

            pressure_solver.assemble_jacobian(time_step);
            pressure_solver.solve();
            pressure_solver.solution += pressure_solver.solution_update;
            std::cout << "Solution increment: "
                      << pressure_solver.solution_update.linfty_norm() << "\t"
                      << std::endl;

          } // end pressure iterations

          // Some info messages (debug)
          std::cout << "Solution limits: "
                    << pressure_solver.solution.linfty_norm() << "\t"
                    << std::endl;
          // std::cout << "Vol strain: "
          //           << volumetric_strain.linfty_norm() << "\t"
          //           << std::endl;

          // Solve displacement system
          displacement_solver.assemble_system(pressure_solver.solution);
          displacement_solver.solve();

          get_normal_strain_components();
          // get_volumetric_strain();

          // get error
          pressure_solver.assemble_residual(time_step,
                                            volumetric_strain,
                                            initial_volumetric_strain);
          pressure_error = pressure_solver.residual.l2_norm();
          std::cout << "        Error: " << pressure_error << std::endl;
        } // end FSS iterations

      get_shear_strain_components();
      get_effective_stresses();
      output_results(time_step_number);

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

  template <int dim>
  void PoroElasticProblem<dim>::refine_mesh(const unsigned int min_grid_level,
                                            const unsigned int max_grid_level)

  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    int fe_degree = pressure_solver.fe.degree;
    KellyErrorEstimator<dim>::estimate (pressure_solver.dof_handler,
                                        QGauss<dim-1>(fe_degree + 1),
                                        typename FunctionMap<dim>::type(),
                                        pressure_solver.solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6, 0.4);
    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag();
    for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag();

    // Pressure solution transfer
    SolutionTransfer<dim> pressure_trans(pressure_solver.dof_handler);
    std::vector< Vector<double> > previous_solution(2);
    previous_solution[0] = pressure_solver.solution;
    previous_solution[1] = volumetric_strain;

    triangulation.prepare_coarsening_and_refinement();
    pressure_trans.prepare_for_coarsening_and_refinement(previous_solution);
    triangulation.execute_coarsening_and_refinement();

    setup_dofs();

    // interpolate old solution
    std::vector< Vector<double> > tmp(3);
    tmp[0].reinit(pressure_solver.solution);
    tmp[1].reinit(volumetric_strain);
    tmp[2].reinit(initial_volumetric_strain);
    pressure_trans.interpolate(previous_solution, tmp);

    // set interpolated solution
    pressure_solver.solution = tmp[0];
    volumetric_strain = tmp[1];
    initial_volumetric_strain = tmp[2];

    // ???
    // pressure_constraints.distribute(pressure_solution);
    // pressure_constraints.distribute(volumetric_strain);
  }

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
