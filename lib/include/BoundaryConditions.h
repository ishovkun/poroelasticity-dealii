#include <deal.II/base/exceptions.h>

namespace boundary_conditions {
  using namespace dealii;

  template <int dim>
  class BoundaryConditions {
  public:
    // BoundaryConditions();
    // ~BoundaryConditions();
    void set_dirichlet(std::vector<int> &labels,
                       std::vector<int> &components,
                       std::vector<double>       &values);
    void set_neumann(std::vector<int> &labels,
                     std::vector<int> &components,
                     std::vector<double>       &values);

    std::vector<int>
      neumann_labels, dirichlet_labels,
      neumann_components, dirichlet_components;
    std::vector<double> dirichlet_values, neumann_values;

    int n_dirichlet;
    int n_neumann;
  };

  template <int dim>
  void BoundaryConditions<dim>::set_dirichlet (
   std::vector<int> &labels,
   std::vector<int> &components,
   std::vector<double>       &values)
  {
    n_dirichlet = labels.size();
    ExcDimensionMismatch(components.size(), n_dirichlet);
    ExcDimensionMismatch(values.size(), n_dirichlet);

    for (int d=0; d<components.size(); ++d)
      Assert(components[d] < dim, ExcNotImplemented());

    dirichlet_labels = labels;
    dirichlet_values = values;
    dirichlet_components = components;
  }

  template <int dim>
  void BoundaryConditions<dim>::set_neumann(
     std::vector<int> &labels,
     std::vector<int> &components,
     std::vector<double>       &values)
  {
    n_neumann = labels.size();
    ExcDimensionMismatch(components.size(), n_neumann);
    ExcDimensionMismatch(values.size(), n_neumann);

    for (int d=0; d<components.size(); ++d){
      Assert(components[d] < dim, ExcNotImplemented());
    }

    neumann_labels = labels;
    neumann_values = values;
    neumann_components = components;
  }
}
