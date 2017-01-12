#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

// custom modules
#include <InputDataPoroel.h>


namespace solvers {
  using namespace dealii;


  template <int dim> class PoroElasticPressureSolver {
  // methods
  public:
    PoroElasticPressureSolver(Triangulation<dim> &triangulation,
                              input_data::InputDataPoroel<dim> &data_,
                              int fe_degree = 1);
    ~PoroElasticPressureSolver();

    void setup_dofs();
    void output_results(const unsigned int time_step_number);
    void assemble_jacobian(double time_step);
    void assemble_residual(double time_step,
                           Vector<double> &volumetric_strain);
    void solve();

  // variables
  public:
    DoFHandler<dim>       dof_handler;
    FE_Q<dim>             fe;
    Vector<double>        solution, solution_update,
                          old_solution;
    Vector<double>        residual;
    Vector<double>        tmp1, tmp2;
    SparsityPattern       sparsity_pattern;
    ConstraintMatrix      constraints;
    SparseMatrix<double>  mass_matrix, laplace_matrix, jacobian;
    input_data::InputDataPoroel<dim> &data;

  };


  template <int dim>
  PoroElasticPressureSolver<dim>::
  PoroElasticPressureSolver(Triangulation<dim> &triangulation,
                            input_data::InputDataPoroel<dim> &data_,
                            int fe_degree) :
  dof_handler(triangulation),
  data(data_),
  fe(fe_degree)
  {}


  template <int dim>
  PoroElasticPressureSolver<dim>::~PoroElasticPressureSolver()
  {
    dof_handler.clear();
  }


  template <int dim>
  void PoroElasticPressureSolver<dim>::setup_dofs()
  {
    { // constraints
      // Just the hanging node constraints because no dirichlet pressure BC's
      dof_handler.distribute_dofs(fe);
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints);
      constraints.close();
    }

    { // assemble sparsity patterns
      unsigned int n_dofs = dof_handler.n_dofs();
      DynamicSparsityPattern dsp(n_dofs);
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      /*keep_constrained_dofs = */ true);
      sparsity_pattern.copy_from(dsp);
     }

    { // initialize vectors and matrices
      unsigned int n_dofs = dof_handler.n_dofs();
      jacobian.reinit(sparsity_pattern);
      mass_matrix.reinit(sparsity_pattern);
      laplace_matrix.reinit(sparsity_pattern);

      MatrixCreator::create_mass_matrix(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        mass_matrix);
      MatrixCreator::create_laplace_matrix(dof_handler,
                                           QGauss<dim>(fe.degree+1),
                                           laplace_matrix);

      solution.reinit(n_dofs);
      old_solution.reinit(n_dofs);
      solution_update.reinit(n_dofs);
      tmp1.reinit(n_dofs);
      tmp2.reinit(n_dofs);
      residual.reinit(n_dofs);
    }

  } // end of method

  template <int dim>
  void PoroElasticPressureSolver<dim>::assemble_residual
    (double time_step,
     Vector<double> &volumetric_strain)
  {
    /* std::cout << "Factors: " */
    /*           << data.biot_coef << "\t" */
    /*           << data.time_step << "\t" */
    /*           << data.perm << "\t" */
    /*           << data.visc << "\t" */
    /*           << data.r_well << "\t" */
    /*           << data.m_modulus << "\t" */
    /*           << std::endl; */
    // Coupling terms
    tmp1 = volumetric_strain;
    /* std::cout << "Pass1" << std::endl; */
    tmp1 *= (data.biot_coef/time_step);
    /* std::cout << "Pass2" << std::endl; */

    // Accumulation term
    tmp2 = solution;
    tmp2 -= old_solution;
    tmp2 *= (1./data.m_modulus/time_step);
    tmp1 += tmp2;
    mass_matrix.vmult(residual, tmp1);

    // Diffusive flow term
    laplace_matrix.vmult(tmp1, solution);
    double factor = data.perm/data.visc;
    tmp1 *= factor;
    residual += tmp1;

    // Source term
    right_hand_side::SinglePhaseWell<dim> source_term_function(data.r_well);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree + 1),
                                        source_term_function,
                                        tmp1);
    residual += tmp1;

    // we are solving jacobian*dp = -residual
    residual *= -1;
    constraints.condense(residual);
  }


  template <int dim>
  void PoroElasticPressureSolver<dim>::assemble_jacobian(double time_step)
  {
    // Accumulation term
    jacobian.copy_from(mass_matrix);
    jacobian *= (1./data.m_modulus/time_step);

    // Diffusive flow term
    double factor = data.perm/data.visc;
    jacobian.add(factor, laplace_matrix);
    constraints.condense(jacobian);
  } // EOM


  template <int dim>
  void PoroElasticPressureSolver<dim>::solve()
  {
    SolverControl solver_control(1000, 1e-8 * residual.l2_norm());
    SolverCG<> cg(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(jacobian, 1.0);
    cg.solve(jacobian, solution_update, residual, preconditioner);
    constraints.distribute(solution_update);
    solution += solution_update;
    // std::cout << "     "
    //           << "Pressure system CG iterations: "
    //           << solver_control.last_step()
    //           << std::endl;
  } // EOM



} // end of namespace
