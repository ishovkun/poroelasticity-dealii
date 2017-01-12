#include <deal.II/base/parameter_handler.h>

namespace input_data {
  using namespace dealii;


  template <int dim> class InputDataPoroel
  {
    // methods
  public:
    InputDataPoroel();
    void read_input_file(std::string input_file_name_);
  private:
    void declare_parameters();
    void assign_parameters();
    void compute_derived_parameters();
    void check_data();

    // variables
  private:
    ParameterHandler prm;
    std::string      input_file_name;
  public:
    // Equation data
    double perm, poro, visc, f_comp;
    double youngs_modulus, poisson_ratio, biot_coef;
    double bulk_density;
    double r_well, flow_rate;
    // Solver control
    double time_step, t_max;
    double fss_tol, pressure_tol;
    int max_fss_iterations, max_pressure_iterations;

    // Derived equation parameters
    double lame_constant, shear_modulus, bulk_modulus,
           grain_bulk_modulus, n_modulus, m_modulus;

  };  // End of class declaration


  template <int dim>
  InputDataPoroel<dim>::InputDataPoroel()
  {}  // EOM


  template <int dim>
    void InputDataPoroel<dim>::read_input_file(std::string input_file_name_)
  {
    input_file_name = input_file_name_;
    declare_parameters();
    prm.read_input(input_file_name);
    prm.print_parameters (std::cout, ParameterHandler::Text);
    assign_parameters();
    compute_derived_parameters();
    check_data();
  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::declare_parameters()
  {
    /* prm.enter_subsection("Mesh"); */
    /* prm.leave_subsection(); */

    {
      prm.enter_subsection("Properties");
      prm.declare_entry("Young modulus", "7e9", Patterns::Double());
      prm.declare_entry("Poisson ratio", "0.3", Patterns::Double());
      prm.declare_entry("Biot coefficient", "0.9", Patterns::Double());
      prm.declare_entry("Permeability", "1", Patterns::Double());
      prm.declare_entry("Porosity", "0.3", Patterns::Double());
      prm.declare_entry("Viscosity", "1e-3", Patterns::Double());
      prm.declare_entry("Bulk density", "2700", Patterns::Double());
      prm.declare_entry("Fluid compressibility", "45.8e-11", Patterns::Double());
      prm.declare_entry("Well radius", "0.1", Patterns::Double());
      prm.declare_entry("Flow rate", "0.1", Patterns::Double());
      prm.leave_subsection();
    }
    {
      prm.enter_subsection("In Situ");
      prm.declare_entry("Pinit", "10e6", Patterns::Double());
      prm.leave_subsection();
    }
    {
      prm.enter_subsection("Solver");
      prm.declare_entry("Time step", "60", Patterns::Double());
      prm.declare_entry("Time max", "60", Patterns::Double());
      prm.declare_entry("Max FSS iterations", "50", Patterns::Integer(3, 1000));
      prm.declare_entry("Max pressure iterations", "50", Patterns::Integer(3, 1000));
      prm.declare_entry("FSS tolerance", "1e-8", Patterns::Double(1e-16, 1e-5));
      prm.declare_entry("Pressure tolerance", "1e-8", Patterns::Double(1e-16, 1e-5));
      prm.leave_subsection();
    }

  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::assign_parameters()
  {
    /* std::cout << "perm: " << perm << std::endl; */
    {
      double mili_darcy = 9.869233e-16;
      prm.enter_subsection("Properties");
      youngs_modulus = prm.get_double("Young modulus");
      poisson_ratio = prm.get_double("Poisson ratio");
      biot_coef = prm.get_double("Biot coefficient");
      perm = prm.get_double("Permeability");
      perm *= mili_darcy;
      poro = prm.get_double("Porosity");
      visc = prm.get_double("Viscosity");
      bulk_density = prm.get_double("Bulk density");
      f_comp = prm.get_double("Fluid compressibility");
      r_well = prm.get_double("Well radius");
      flow_rate = prm.get_double("Flow rate");
      prm.leave_subsection();
    }
    {
      prm.enter_subsection("Solver");
      time_step = prm.get_double("Time step");
      t_max = prm.get_double("Time max");
      fss_tol = prm.get_double("FSS tolerance");
      pressure_tol = prm.get_double("Pressure tolerance");
      max_fss_iterations = prm.get_integer("Max FSS iterations");
      max_pressure_iterations = prm.get_integer("Max pressure iterations");
      prm.leave_subsection();
    }
    /* std::cout << "perm: " << perm << std::endl; */
  }  // EOM


  template <int dim>
  void InputDataPoroel<dim>::compute_derived_parameters()
  {
    double E = youngs_modulus, nu = poisson_ratio;
    lame_constant = E*nu/((1.+nu)*(1.-2.*nu));
    shear_modulus = 0.5*E/(1+nu);
    bulk_modulus = lame_constant + 2./3.*shear_modulus;
    grain_bulk_modulus = bulk_modulus/(1 - biot_coef);
    n_modulus = grain_bulk_modulus/(biot_coef - poro);
    m_modulus = (n_modulus/f_comp)/(n_modulus*poro + 1./f_comp);
  }  // EOM


  template <int dim>
    void InputDataPoroel<dim>::check_data()
    {
      /* std::cout << "Params: " << std::endl */
      /*           << "f_comp " << f_comp << std::endl */
      /*           << "perm " << perm << std::endl */
      /*           << "lambda " << lame_constant << std::endl */
      /*           << "G " << shear_modulus << std::endl */
      /*           << "K " << bulk_modulus << std::endl */
      /*           << "Ks " << grain_bulk_modulus << std::endl */
      /*           << "N " << n_modulus << std::endl */
      /*           << "M " << m_modulus << std::endl; */

    }  // EOM

}  // end of namespace
