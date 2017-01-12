#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

// custom modules


namespace projection {
  using namespace dealii;


  template <int dim>
  class StrainProjector {
  /*
    Class that projects strains from displacement solution space to
    pressure solution space
  */
  // methods
  public:
    StrainProjector();
    void set_solvers(solvers::PoroElasticDisplacementSolver<dim> &displacement_solver,
                     solvers::PoroElasticPressureSolver<dim> &pressure_solver);
    void setup_dofs();
    void assemble_projection_matrix();
    void assemble_projection_rhs(std::vector<int> tensor_components);
    void solve_projection_system(int rhs_entry);


  // variables
  private:
    solvers::PoroElasticDisplacementSolver<dim>  *p_displacement_solver;
    solvers::PoroElasticPressureSolver<dim>      *p_pressure_solver;

    int                   n_stress_components;
    std::vector<int>      strain_tensor_volumetric_components,
                          strain_rhs_volumetric_entries,
                          strain_tensor_shear_components;

    indexing::TensorIndexer<dim>    tensor_indexer;

  public:
    std::vector< Vector<double> > projection_rhs, strains;
    SparseMatrix<double>  projection_matrix;


  }; // end of class definition


  template <int dim>
  StrainProjector<dim>::StrainProjector()
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
  }  // EOM


  template <int dim>
  void StrainProjector<dim>::set_solvers
    (solvers::PoroElasticDisplacementSolver<dim> &displacement_solver_,
     solvers::PoroElasticPressureSolver<dim> &pressure_solver_)
   {
     p_displacement_solver = &displacement_solver_;
     p_pressure_solver = &pressure_solver_;
   }   // EOM


  template <int dim>
  void StrainProjector<dim>::setup_dofs()
  {
    projection_rhs.resize(n_stress_components);
    projection_matrix.reinit(p_pressure_solver->sparsity_pattern);
    strains.resize(n_stress_components);

    DoFHandler<dim>  *p_dof_handler = &(p_pressure_solver->dof_handler);
    unsigned int n_dofs = p_dof_handler->n_dofs();
    /* std::cout << "ndofs: " << n_dofs << std::endl; */

    for (int i=0; i<n_stress_components; ++i){
      projection_rhs[i].reinit(n_dofs);
      strains[i].reinit(n_dofs);
    }

  }  // EOM


  template <int dim>
  void StrainProjector<dim>::assemble_projection_matrix()
    {
      projection_matrix.copy_from(p_pressure_solver->mass_matrix);
      p_pressure_solver->constraints.condense(projection_matrix);
    }


  template <int dim>
  void StrainProjector<dim>::
  assemble_projection_rhs(std::vector<int> tensor_components)
  {
    // check input
    int n_comp = tensor_components.size();

    // put assert statements

    // access pressure/displacement fe
    FE_Q<dim> &pressure_fe = p_pressure_solver->fe;
    FESystem<dim> &displacement_fe = p_displacement_solver->fe;

    // access pressure/displacement dofs
    DoFHandler<dim> &pressure_dof_handler = p_pressure_solver->dof_handler;
    DoFHandler<dim> &displacement_dof_handler = p_displacement_solver->dof_handler;

	  QGauss<dim>  quadrature_formula(pressure_fe.degree+1);
	  FEValues<dim> pressure_fe_values(pressure_fe, quadrature_formula,
                                     update_values |
                                     update_quadrature_points |
                                     update_JxW_values);

	  FEValues<dim> displacement_fe_values(displacement_fe, quadrature_formula,
                                         update_gradients |
                                         update_quadrature_points);

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = pressure_fe.dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // global vectors where to write the RHS values
    std::vector<int> entries = tensor_indexer.entryIndex(tensor_components);
    std::vector< Vector<double> > cell_rhs(n_comp, Vector<double>(dofs_per_cell));

    // fill global vectors with zeros
    for (int c=0; c<n_comp; ++c)
      projection_rhs[entries[c]] = 0;

	  SymmetricTensor<2, dim> strain_tensor;
    std::vector< std::vector<Tensor<1,dim> > >
      displacement_grads(quadrature_formula.size(),
                         std::vector<Tensor<1,dim> >(dim));

    typename DoFHandler<dim>::active_cell_iterator
      cell = pressure_dof_handler.begin_active(),
      endc = pressure_dof_handler.end(),
      displacement_cell = displacement_dof_handler.begin_active();

    for (; cell!=endc; ++cell, ++displacement_cell) {
      // reinit stuff -------
      for (int c=0; c<n_comp; ++c) cell_rhs[c] = 0;
		  pressure_fe_values.reinit(cell);
		  displacement_fe_values.reinit(displacement_cell);
      displacement_fe_values.get_function_gradients(p_displacement_solver->solution,
                                                    displacement_grads);
      // ----------

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
        strain_tensor = constitutive_model::get_strain_tensor(displacement_grads[q_point]);
        double jxw = pressure_fe_values.JxW(q_point);


        for (int i=0; i<dofs_per_cell; ++i){
          double phi_i = pressure_fe_values.shape_value(i, q_point);

          for (int c=0; c<n_comp; ++c) {
            int comp = tensor_components[c];
            int tensor_component_1 = comp/dim;
            int tensor_component_2 = comp%dim;
            double strain_value =
              strain_tensor[tensor_component_1][tensor_component_2];
            // std::cout << strain_value << std::endl;

            cell_rhs[c][i] += (phi_i * strain_value * jxw);
          }
        }  // end of i loop

      }  // end of q_point loop

      // distribute local values to global vectors
      cell->get_dof_indices(local_dof_indices);
      for (int c=0; c<n_comp; ++c)
        p_pressure_solver->constraints.distribute_local_to_global
          (cell_rhs[c], local_dof_indices, projection_rhs[entries[c]]);

    }  // end of cell iteration

  }  // EOM


  template <int dim>
  void StrainProjector<dim>::solve_projection_system(int rhs_entry)
  {
    // select entries
    auto& rhs_vector = projection_rhs[rhs_entry];
    auto& solution_vector = strains[rhs_entry];

    // solve
    SolverControl solver_control(1000, 1e-8 * rhs_vector.l2_norm());
    SolverCG<> cg(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(projection_matrix, 1.0);
    cg.solve(projection_matrix, solution_vector, rhs_vector,
             preconditioner);
    p_pressure_solver->constraints.distribute(solution_vector);

    // std::cout << "     "
    //           << "Projection component: "
    //           << rhs_entry
    //           << " CG iterations: "
    //           << solver_control.last_step()
    //           << std::endl;
    // std::cout << "     "
    //           << "RHS norm "
    //           << rhs_vector.l2_norm()
    //           << " solution: "
    //           << solution_vector.l2_norm()
    //           << " solution1: "
    //           << strains[rhs_entry].l2_norm()
    //           << std::endl
    //           << std::endl;
  } // EOM

  }  // end of namespace
