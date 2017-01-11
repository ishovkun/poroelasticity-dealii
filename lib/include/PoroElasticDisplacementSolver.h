#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

// custom modules
#include <BoundaryConditions.h>
#include <ConstitutiveModel.h>

namespace solvers {
  using namespace dealii;


  template <int dim> class PoroElasticDisplacementSolver {
  // methods
  public:
    PoroElasticDisplacementSolver(Triangulation<dim> &triangulation,
                                  int fe_degree = 2);
    ~PoroElasticDisplacementSolver();

    void setup_dofs();
    void set_pressure_fe(DoFHandler<dim> &pressure_dof_handler_,
                         FE_Q<dim> &pressure_fe_);

    void assemble_system(Vector<double> &pressure_solution);
    void set_boundary_conditions(
      std::vector<unsigned int> neumann_labels,
      std::vector<unsigned int> neumann_components,
      std::vector<double>       neumann_values,
      std::vector<unsigned int> dirichlet_labels,
      std::vector<unsigned int> dirichlet_components,
      std::vector<double>       dirichlet_values
    );

    void solve();

  // variables
  public:
    DoFHandler<dim>       dof_handler;
    FESystem<dim>         fe;
    // we use pointers for these two variables, because references require
    // initialization. the variables are initialized in a different class
    DoFHandler<dim>       *p_pressure_dof_handler;
    FE_Q<dim>             *p_pressure_fe;
    Vector<double>        solution;

  private:
    ConstraintMatrix      constraints;
    SparsityPattern       sparsity_pattern;
    SparseMatrix<double>  system_matrix;
    Vector<double>        rhs_vector;
    boundary_conditions::BoundaryConditions<dim> bc;
    bool rebuild_system_matrix;

  };


  template <int dim>
  PoroElasticDisplacementSolver<dim>::
  PoroElasticDisplacementSolver(Triangulation<dim> &triangulation,
                                int fe_degree) :
    dof_handler(triangulation),
    fe(FE_Q<dim>(2), dim)
    {}


  template <int dim>
  PoroElasticDisplacementSolver<dim>::~PoroElasticDisplacementSolver()
  {
    dof_handler.clear();
  }


  template <int dim>
  void PoroElasticDisplacementSolver<dim>::set_boundary_conditions(
    std::vector<unsigned int> neumann_labels,
    std::vector<unsigned int> neumann_components,
    std::vector<double>       neumann_values,
    std::vector<unsigned int> dirichlet_labels,
    std::vector<unsigned int> dirichlet_components,
    std::vector<double>       dirichlet_values)
  {
    bc.set_dirichlet (dirichlet_labels,
                      dirichlet_components,
                      dirichlet_values);

    bc.set_neumann (neumann_labels,
                    neumann_components,
                    neumann_values);
  }

  template <int dim>
    void PoroElasticDisplacementSolver<dim>::set_pressure_fe
    (DoFHandler<dim> &pressure_dof_handler_,
     FE_Q<dim> &pressure_fe_)
    {
      p_pressure_fe = &pressure_fe_;
      p_pressure_dof_handler = &pressure_dof_handler_;
    }


  template <int dim>
    void PoroElasticDisplacementSolver<dim>::setup_dofs()
  {
    { // constrains
      dof_handler.distribute_dofs(fe);

      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints);

      // apply dirichlet BC's
      std::vector<ComponentMask> mask(dim);
      for (unsigned int comp=0; comp<dim; ++comp){
        FEValuesExtractors::Scalar extractor(comp);
        mask[comp] = fe.component_mask(extractor);
      }

      int n_dirichlet_conditions = bc.dirichlet_labels.size();

      for (int cond=0; cond<n_dirichlet_conditions; ++cond) {
           int component = bc.dirichlet_components[cond];
           double dirichlet_value = bc.dirichlet_values[cond];
           VectorTools::interpolate_boundary_values
             (dof_handler,
              bc.dirichlet_labels[cond],
              ConstantFunction<dim>(dirichlet_value, dim),
              constraints,
              mask[component]);
      }

      constraints.close();
      rebuild_system_matrix = true;
    }

    { // create sparsity patterns, init vectors and matrices
      unsigned int n_dofs = dof_handler.n_dofs();
      DynamicSparsityPattern dsp(n_dofs);
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp, constraints,
                                      /*keep_constrained_dofs = */ true);
      sparsity_pattern.copy_from(dsp);

      // matrices & vectors
      system_matrix.reinit(sparsity_pattern);
      rhs_vector.reinit(n_dofs);
      solution.reinit(n_dofs);
    }
  }

  template <int dim>
  void PoroElasticDisplacementSolver<dim>::assemble_system
    (Vector<double> &pressure_solution)
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim-1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    FEValues<dim> pressure_fe_values(*p_pressure_fe, quadrature_formula,
                                     update_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values |
                                     update_quadrature_points |
                                     update_normal_vectors |
                                     update_JxW_values);

    double bulk_density = 2700;
    double lame_constant = 1e5, shear_modulus = 1e6;
    double biot_coef = 0.8;
    right_hand_side::BodyForces<dim>  body_force(bulk_density);

    // fe parameters
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // local vectors & matrices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double>      cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>          cell_rhs(dofs_per_cell);
    std::vector< Vector<double> > rhs_values(n_q_points,
                                             Vector<double>(dim));
    Tensor<1,dim>	neumann_bc_vector;
    SymmetricTensor<2,dim>	strain_tensor_i, strain_tensor_j;
    SymmetricTensor<4,dim>  gassman_tensor =
      constitutive_model::isotropic_gassman_tensor<dim>(lame_constant, shear_modulus);

    // store pressure values
    std::vector<double> pressure_values(n_q_points);
    unsigned int n_neumann_conditions = bc.neumann_labels.size();

    // iterators
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end(),
      pressure_cell = p_pressure_dof_handler->begin_active();

    rhs_vector = 0;

    for (; cell!=endc; ++cell, ++pressure_cell) {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);
      pressure_fe_values.reinit(pressure_cell);
      pressure_fe_values.get_function_values(pressure_solution,
                                             pressure_values);
      body_force.vector_value_list
        (fe_values.get_quadrature_points(), rhs_values);

      for (unsigned int i=0; i<dofs_per_cell; ++i){
        const unsigned int
          component_i = fe.system_to_component_index(i).first;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
          double jxw = fe_values.JxW(q_point);
          // body forces
          cell_rhs(i) +=
            (fe_values.shape_value(i, q_point) *
             rhs_values[q_point](component_i)
            ) * jxw;

          // Pore pressure coupling
          // it only has one non-negative diagonal entry
          // should be rewritte for gradint(conmonent_i)
          strain_tensor_i =
            constitutive_model::get_strain_tensor(fe_values, i, q_point);
          cell_rhs(i) +=
          (biot_coef*pressure_values[q_point] *
           trace(strain_tensor_i)) * jxw;


          for (unsigned int j=0; j<dofs_per_cell; ++j){
            strain_tensor_j =
              constitutive_model::get_strain_tensor(fe_values, j, q_point);
            cell_matrix(i, j) +=
              (gassman_tensor*strain_tensor_i*strain_tensor_j*jxw);
          }

        } // end loop over q_points

      }  // end loop over i

      // impose neumann BC's
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f){
        if (cell->face(f)->at_boundary()) {
          unsigned int face_boundary_id = cell->face(f)->boundary_id();
          fe_face_values.reinit(cell, f);

          // loop through different boundary labels
          for (unsigned int l=0; l<n_neumann_conditions; ++l){
            int id = bc.neumann_labels[l];

            if (face_boundary_id == id)
              for (unsigned int i=0; i<dofs_per_cell; ++i){
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;

                if (component_i == bc.neumann_components[l])
                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point) {
                    double neumann_value =
                      bc.neumann_values[l] *
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

      cell->get_dof_indices(local_dof_indices);
      if (rebuild_system_matrix)
        constraints.distribute_local_to_global
          (cell_matrix, cell_rhs, local_dof_indices,
           system_matrix, rhs_vector);
      else
      constraints.distribute_local_to_global
        (cell_rhs, local_dof_indices, rhs_vector, cell_matrix);

    } // end of cell loop

    rebuild_system_matrix = false;
  } // EOM


  template <int dim>
  void PoroElasticDisplacementSolver<dim>::solve()
  {
    // double solver_tolerance = 1e-8*displacement_rhs.l2_norm();
    double         solver_tolerance = 1e-12;
    SolverControl  solver_control(1000, solver_tolerance);
    SolverCG<>     cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, rhs_vector, preconditioner);
    constraints.distribute(solution);
  } // EOM



} // end of namespace
