import matplotlib.pyplot as plt
import h5py
from ultranest.netiter import BreadthFirstIterator, MultiCounter, combine_results, PointPile, TreeNode
import numpy as np
import sys
import os
import joblib

mem = joblib.Memory('.', verbose=False)

class TrackingUniformOrderAccumulator():
    """U test accumulator.

    Stores rank orders (1 to N), for comparison with a uniform order.

    See section 4.5.2 of Buchner (2023, https://arxiv.org/abs/2101.09675),
    based on the Mann-Whitney-Wilcoxon U test against a uniform integer
    distribution.
    """

    def __init__(self):
        """Initiate empty accumulator."""
        self.N = 0
        self.U = 0.0
        self.orders = []
        self.Ns = []

    def reset(self):
        """Set all counts to zero."""
        self.N = 0
        self.U = 0.0

    def add(self, order, N):
        """
        Accumulate rank *order* (0 to N).

        Parameters
        ----------
        N: int
            maximum rank allowed.
        order: int
            rank between 0 and N (inclusive).
        """
        if not 0 <= order <= N:
            raise ValueError("order %d out of %d invalid" % (order, N))
        self.U += (order + 0.5) / N
        self.N += 1
        self.orders.append(order)
        self.Ns.append(N)

    @property
    def zscore(self):
        """z-score of the null hypothesis (uniform distribution) probability."""
        N = self.N
        if N == 0:
            return 0.0
        m_U = N * 0.5
        sigma_U_corr = (N / 12.0)**0.5
        return (self.U - m_U) / sigma_U_corr

    def __len__(self):
        """Return number of samples accumulated so far."""
        return self.N


def logz_sequence(root, pointpile, nbootstraps=12, random=True, onNode=None, verbose=False, check_insertion_order=True):
    """Run MultiCounter through tree `root`.

    Keeps track of, and returns ``(logz, logzerr, logv, nlive)``.

    Parameters
    ----------
    root: :py:class:`TreeNode`
        Tree
    pointpile: :py:class:`PointPile`
        Point pile
    nbootstraps: int
        Number of independent iterators
    random: bool
        Whether to randomly draw volume estimates
    onNode: function
        Function to call for every node.
        receives current node and the iterator
    verbose: bool
        Whether to show a progress indicator on stderr
    check_insertion_order: bool
        Whether to perform a rolling insertion order rank test

    Returns
    --------
    results: dict
        Run information, see :py:func:`combine_results`
    sequence: dict
        Each entry of the dictionary is results['niter'] long,
        and contains the state of information at that iteration.
        Important keys are:
        Iteration number (niter),
        Volume estimate (logvol), loglikelihood (logl), absolute logarithmic weight (logwt),
        Relative weight (weights), point (samples),
        Number of live points (nlive),
        Evidence estimate (logz) and its uncertainty (logzerr),
        Rank test score (insert_order).

    """
    roots = root.children

    Lmax = -np.inf

    explorer = BreadthFirstIterator(roots)
    # Integrating thing
    main_iterator = MultiCounter(
        nroots=len(roots), nbootstraps=max(1, nbootstraps),
        random=random, check_insertion_order=check_insertion_order)
    main_iterator.insertion_order_accumulator = TrackingUniformOrderAccumulator()
    main_iterator.Lmax = max(Lmax, max(n.value for n in roots))

    logz = []
    logzerr = []
    nlive = []
    logvol = []
    niter = 0

    saved_nodeids = []
    saved_logl = []
    insert_order = []
    # we go through each live point (regardless of root) by likelihood value
    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next_node
        # this is the likelihood level we have to improve upon
        Lmin = node.value

        if onNode:
            onNode(node, main_iterator)

        logz.append(main_iterator.logZ)
        with np.errstate(invalid='ignore'):
            # first time they are all the same
            logzerr.append(main_iterator.logZerr_bs)

        nactive = len(active_values)

        if len(np.unique(active_values)) == nactive and len(node.children) > 0:
            child_insertion_order = (active_values > node.children[0].value).sum()
            insert_order.append(2 * (child_insertion_order + 1.) / nactive)
        else:
            insert_order.append(np.nan)

        nlive.append(nactive)
        logvol.append(main_iterator.logVolremaining)

        niter += 1
        if verbose:
            sys.stderr.write("%d...\r" % niter)

        saved_logl.append(Lmin)
        saved_nodeids.append(node.id)
        # inform iterators (if it is their business) about the arc
        main_iterator.passing_node(rootid, node, active_rootids, active_values)
        explorer.expand_children_of(rootid, node)

    logwt = np.asarray(saved_logl) + np.asarray(main_iterator.logweights)[:,0]
    logvol[-1] = logvol[-2]

    results = combine_results(saved_logl, saved_nodeids, pointpile, main_iterator)
    sequence = dict(
        logz=np.asarray(logz),
        logzerr=np.asarray(logzerr),
        logvol=np.asarray(logvol),
        samples_n=np.asarray(nlive),
        nlive=np.asarray(nlive),
        insert_order=np.asarray(insert_order),
        logwt=logwt,
        niter=niter,
        logl=saved_logl,
        weights=results['weighted_samples']['weights'],
        samples=results['weighted_samples']['points'],
        insert_orders = np.array(main_iterator.insertion_order_accumulator.orders),
        insert_Ns = np.array(main_iterator.insertion_order_accumulator.Ns),
    )

    return sequence, results

@mem.cache
def getrunqual(log_dir, x_dim, num_bootstraps=20, random=True, verbose=False, check_insertion_order=True):
    filepath = os.path.join(log_dir, 'results', 'points.hdf5')
    fileobj = h5py.File(filepath, 'r')
    _, ncols = fileobj['points'].shape
    num_params = ncols - 3 - x_dim

    points = fileobj['points'][:]
    fileobj.close()
    del fileobj
    stack = list(enumerate(points))

    pointpile = PointPile(x_dim, num_params)

    def pop(Lmin):
        """Find matching sample from points file."""
        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        for i, (idx, next_row) in enumerate(stack):
            row_Lmin = next_row[0]
            L = next_row[1]
            if row_Lmin <= Lmin and L > Lmin:
                idx, row = stack.pop(i)
                return idx, row
        return None, None

    roots = []
    while True:
        _, row = pop(-np.inf)
        if row is None:
            break
        logl = row[1]
        u = row[3:3 + x_dim]
        v = row[3 + x_dim:3 + x_dim + num_params]
        roots.append(pointpile.make_node(logl, u, v))

    root = TreeNode(id=-1, value=-np.inf, children=roots)

    def onNode(node, main_iterator):
        """Insert (single) child of node if available."""
        while True:
            _, row = pop(node.value)
            if row is None:
                break
            if row is not None:
                logl = row[1]
                u = row[3:3 + x_dim]
                v = row[3 + x_dim:3 + x_dim + num_params]
                child = pointpile.make_node(logl, u, v)
                assert logl > node.value, (logl, node.value)
                main_iterator.Lmax = max(main_iterator.Lmax, logl)
                node.children.append(child)

    return logz_sequence(root, pointpile, nbootstraps=num_bootstraps,
                         random=random, onNode=onNode, verbose=verbose,
                         check_insertion_order=check_insertion_order)


plt.figure(figsize=(10, 3))
x_dim = int(sys.argv[1])
prefix = sys.argv[2]
nsteps0 = int(sys.argv[3])
nsteps = nsteps0
ls = ['-'] * 7 + ['--'] * 7
while os.path.exists(prefix % nsteps):
	log_dir = prefix % nsteps
	print("reading %s ..." % log_dir)
	sequence, results = getrunqual(log_dir, x_dim)
	plt.hist(sequence['insert_orders'], label=nsteps, histtype='step', 
		density=True, 
		bins=400, cumulative=True, lw=0.2, ls=ls.pop(0)
		)
	#plt.hist(results['insert_orders'])
	nsteps *= 2
	#break

plt.ylabel('Cumulative frequency')
plt.xlabel('Insertion order')
plt.legend(title='nsteps')
plt.savefig(log_dir + '/plots/ordertest.pdf', bbox_inches='tight')
plt.close()
