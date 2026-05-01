#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Sparse>

extern "C" {
#include "osqp.h"
}

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"

// ─── constants ───────────────────────────────────────────────────────────────
static constexpr int    NXK        = 4;
static constexpr int    NU         = 2;
static constexpr int    TK         = 8;
static constexpr double DTK        = 0.1;
static constexpr double WB         = 0.33;
static constexpr double MAX_STEER  =  0.4189;
static constexpr double MIN_STEER  = -0.4189;
static constexpr double MAX_DSTEER = M_PI;      // 180 deg/s
static constexpr double MAX_SPEED  =  6.0;
static constexpr double MIN_SPEED  =  0.0;
static constexpr double MAX_ACCEL  =  3.0;
static constexpr int    INTERP_N   = 10000;

// decision-variable sizes
static constexpr int NW   = NXK * (TK + 1) + NU * TK;  // 52
static constexpr int NCON = NXK * TK + NXK + (TK + 1) + TK + TK + (TK - 1); // 68

// cost matrix diagonals
static const Eigen::Vector4d Qd  {13.5, 13.5, 13.0, 13.0};
static const Eigen::Vector4d Qfd {13.5, 13.5, 13.0, 13.0};
static const Eigen::Vector2d Rd  { 0.01, 100.0};
static const Eigen::Vector2d Rdd { 0.01, 100.0};

// ─── utilities ───────────────────────────────────────────────────────────────
static double wrap_pi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

// nearest point search — same logic as utils.py nearest_point
static int nearest_index(double px, double py,
                         const std::vector<double>& cx,
                         const std::vector<double>& cy)
{
    int N = (int)cx.size();
    double best = std::numeric_limits<double>::max();
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        double dx = px - cx[i], dy = py - cy[i];
        double d = dx*dx + dy*dy;
        if (d < best) { best = d; idx = i; }
    }
    return idx;
}

// ─── model linearisation  ─────────────────────────────────────────────────
struct ModelMatrices {
    Eigen::Matrix4d A;
    Eigen::Matrix<double,4,2> B;
    Eigen::Vector4d C;
};

static ModelMatrices get_model_matrix(double v, double phi, double delta)
{
    ModelMatrices m;
    m.A = Eigen::Matrix4d::Identity();
    m.A(0, 2) =  DTK * std::cos(phi);
    m.A(0, 3) = -DTK * v * std::sin(phi);
    m.A(1, 2) =  DTK * std::sin(phi);
    m.A(1, 3) =  DTK * v * std::cos(phi);
    m.A(3, 2) =  DTK * std::tan(delta) / WB;

    m.B.setZero();
    m.B(2, 0) = DTK;
    double cd2 = std::cos(delta) * std::cos(delta);
    m.B(3, 1) = DTK * v / (WB * cd2);

    m.C(0) =  DTK * v * std::sin(phi) * phi;
    m.C(1) = -DTK * v * std::cos(phi) * phi;
    m.C(2) =  0.0;
    m.C(3) = -DTK * v * delta / (WB * cd2);
    return m;
}

// ─── MPC node ────────────────────────────────────────────────────────────────
class MPC : public rclcpp::Node
{
public:
    MPC() : Node("mpc_node")
    {
        drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        pose_sub_  = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/pf/viz/inferred_pose", 1,
            std::bind(&MPC::pose_callback, this, std::placeholders::_1));
        odom_sub_  = create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 1,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                last_v_ = msg->twist.twist.linear.x;
            });

        marker_pub_   = create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker", 10);
        cur_goal_pub_ = create_publisher<visualization_msgs::msg::Marker>("/cur_marker", 10);
        pred_pub_     = create_publisher<visualization_msgs::msg::Marker>("/mpc_pred_traj", 10);
        ref_pub_      = create_publisher<visualization_msgs::msg::Marker>("/mpc_ref_traj", 10);
        wp_timer_     = create_wall_timer(std::chrono::milliseconds(500),
                                         std::bind(&MPC::publish_all_waypoints, this));

        load_waypoints("/home/team10/f1tenth_ws/path_racePenn.csv");
        build_qp();

        oa_.assign(TK, 0.0);
        od_.assign(TK, 0.0);

        RCLCPP_INFO(get_logger(), "MPC initialized");
    }

    ~MPC() {
        if (solver_) osqp_cleanup(solver_);
    }

private:
    // ── publishers / subscribers ────────────────────────────────────────────
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    double last_v_{0.0};
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cur_goal_pub_, pred_pub_, ref_pub_;
    rclcpp::TimerBase::SharedPtr wp_timer_;

    // ── waypoints ───────────────────────────────────────────────────────────
    std::vector<double> wp_x_, wp_y_, wp_yaw_, wp_v_;
    double dlk_{0.03};

    // ── OSQP ────────────────────────────────────────────────────────────────
    OSQPSolver*  solver_{nullptr};

    // CSC data for P (constant)
    std::vector<double>    P_x_;
    std::vector<OSQPInt>   P_i_, P_p_;

    // CSC data for A (partially dynamic — dynamics rows update each solve)
    std::vector<double>    A_x_;
    std::vector<OSQPInt>   A_i_, A_p_;

    // We keep track of where the A[t] and B[t] values live in A_x_
    // for fast update each solve.
    // Row layout: [dyn(32), init(4), speed(9), accel(8), steer(8), dsteer(7)]
    // Each dynamics block for step t contributes columns from xk[:,t], xk[:,t+1], uk[:,t]
    // We store per-step index maps for A_x_ updates.
    struct DynIdx {
        // Indices in A_x_ for A[t] (16 entries, row-major within the 4x4 block)
        // and B[t] (8 entries).
        // Note: CSC is column-major so we store indices by their position in A_x_.
        std::array<int,16> Aidx;  // A matrix 4x4 = 16 entries
        std::array<int,8>  Bidx;  // B matrix 4x2 = 8 entries
        std::array<int,4>  Cidx;  // C vector for l/u bounds
    };
    std::vector<DynIdx> dyn_idx_;

    // bounds vectors (kept alive for osqp update)
    std::vector<double> q_vec_, l_vec_, u_vec_;

    // warm-start storage
    std::vector<double> w_prev_, y_prev_;

    // previous solution
    std::vector<double> oa_, od_;
    bool init_flag_{false};

    // ── row offsets in A ─────────────────────────────────────────────────────
    // dyn_row0  = 0
    // init_row0 = NXK*TK       = 32
    // speed_row0= NXK*TK+NXK   = 36
    // accel_row0= NXK*TK+NXK+(TK+1) = 45
    // steer_row0= accel_row0+TK = 53
    // dst_row0  = steer_row0+TK = 61
    static constexpr int ROW_DYN   = 0;
    static constexpr int ROW_INIT  = NXK * TK;                    // 32
    static constexpr int ROW_SPEED = ROW_INIT + NXK;              // 36
    static constexpr int ROW_ACCEL = ROW_SPEED + (TK + 1);        // 45
    static constexpr int ROW_STEER = ROW_ACCEL + TK;              // 53
    static constexpr int ROW_DSTR  = ROW_STEER + TK;              // 61

    // ── waypoint loading ─────────────────────────────────────────────────────
    void load_waypoints(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open waypoint file: %s", path.c_str());
            return;
        }
        std::vector<double> raw_x, raw_y;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream ss(line);
            double x, y;
            if (ss >> x >> y) { raw_x.push_back(x); raw_y.push_back(y); }
        }
        int N = (int)raw_x.size();
        if (N < 2) return;

        // chord-length parameterisation
        std::vector<double> s(N, 0.0);
        for (int i = 1; i < N; ++i) {
            double dx = raw_x[i] - raw_x[i-1], dy = raw_y[i] - raw_y[i-1];
            s[i] = s[i-1] + std::sqrt(dx*dx + dy*dy);
        }
        double total = s.back();

        // linear interpolation to INTERP_N points (closed loop)
        wp_x_.resize(INTERP_N); wp_y_.resize(INTERP_N);
        for (int k = 0; k < INTERP_N; ++k) {
            double tk = total * k / INTERP_N;
            // binary search in s
            auto it = std::lower_bound(s.begin(), s.end(), tk);
            int hi = (int)std::distance(s.begin(), it);
            if (hi == 0) hi = 1;
            if (hi >= N) hi = N - 1;
            int lo = hi - 1;
            double frac = (tk - s[lo]) / (s[hi] - s[lo] + 1e-12);
            wp_x_[k] = raw_x[lo] + frac * (raw_x[hi] - raw_x[lo]);
            wp_y_[k] = raw_y[lo] + frac * (raw_y[hi] - raw_y[lo]);
        }

        // mean step distance
        double sum_ds = 0.0;
        for (int k = 0; k < INTERP_N; ++k) {
            int k1 = (k + 1) % INTERP_N;
            double dx = wp_x_[k1] - wp_x_[k], dy = wp_y_[k1] - wp_y_[k];
            sum_ds += std::sqrt(dx*dx + dy*dy);
        }
        dlk_ = sum_ds / INTERP_N;

        // yaw
        wp_yaw_.resize(INTERP_N);
        for (int k = 0; k < INTERP_N; ++k) {
            int k1 = (k + 1) % INTERP_N;
            wp_yaw_[k] = std::atan2(wp_y_[k1] - wp_y_[k], wp_x_[k1] - wp_x_[k]);
        }

        // curvature-adaptive speed
        wp_v_.resize(INTERP_N);
        double curv_max = 0.0;
        std::vector<double> curvature(INTERP_N);
        for (int k = 0; k < INTERP_N; ++k) {
            int k1 = (k + 1) % INTERP_N;
            double dyaw = wrap_pi(wp_yaw_[k1] - wp_yaw_[k]);
            curvature[k] = std::abs(dyaw) / (dlk_ + 1e-12);
            curv_max = std::max(curv_max, curvature[k]);
        }
        curv_max += 1e-6;
        const double VMIN = 0.8, VMAX = 2.0;
        for (int k = 0; k < INTERP_N; ++k) {
            double cn = std::min(curvature[k] / curv_max, 1.0);
            wp_v_[k] = VMAX - (VMAX - VMIN) * cn;
        }
        RCLCPP_INFO(get_logger(), "Loaded %d waypoints, dlk=%.4f", INTERP_N, dlk_);
    }

    // ── reference trajectory ─────────────────────────────────────────────────
    // Returns (NXK x (TK+1)) column-major reference, wrapped in a flat vector
    void calc_ref_trajectory(double sx, double sy, double sv, double syaw,
                             Eigen::Matrix<double, NXK, TK+1>& ref) const
    {
        int nc = (int)wp_x_.size();
        int ind = nearest_index(sx, sy, wp_x_, wp_y_);
        publish_cur_waypoint_impl(wp_x_[ind], wp_y_[ind]);

        double travel = std::abs(sv) * DTK;
        double dind   = travel / dlk_;

        for (int j = 0; j <= TK; ++j) {
            int i = (ind + (int)(j * dind)) % nc;
            ref(0, j) = wp_x_[i];
            ref(1, j) = wp_y_[i];
            ref(2, j) = wp_v_[i];
            double yaw = wp_yaw_[i];
            // yaw wrap
            while (yaw - syaw >  4.5) yaw -= 2*M_PI;
            while (yaw - syaw < -4.5) yaw += 2*M_PI;
            ref(3, j) = yaw;
        }
    }

    // ── motion prediction ────────────────────────────────────────────────────
    Eigen::Matrix<double, NXK, TK+1>
    predict_motion(const Eigen::Vector4d& x0,
                   const std::vector<double>& oa,
                   const std::vector<double>& od) const
    {
        Eigen::Matrix<double, NXK, TK+1> pp;
        pp.col(0) = x0;
        double x = x0(0), y = x0(1), v = x0(2), yaw = x0(3);
        for (int t = 0; t < TK; ++t) {
            double delta = std::clamp(od[t], MIN_STEER, MAX_STEER);
            x   += v * std::cos(yaw) * DTK;
            y   += v * std::sin(yaw) * DTK;
            yaw += (v / WB) * std::tan(delta) * DTK;
            v   += oa[t] * DTK;
            v    = std::clamp(v, MIN_SPEED, MAX_SPEED);
            pp(0,t+1) = x; pp(1,t+1) = y;
            pp(2,t+1) = v; pp(3,t+1) = yaw;
        }
        return pp;
    }

    // ── QP build (called once at startup) ────────────────────────────────────
    void build_qp()
    {
        // ── P matrix (constant, 52x52, upper triangular) ─────────────────────
        // P = 2 * block_diag(Qblock_36x36, (Rblock + D'*Rdblock*D)_16x16)
        // Qblock = diag(Q,Q,...,Q,Qf), all identical here.
        // Since Q and R are diagonal, P is diagonal too except for the
        // consistency term D'*Rdblock*D.

        // Build Puv = Rblock + D'*Rdblock*D  (16x16)
        Eigen::MatrixXd Rblock = Eigen::MatrixXd::Zero(NU*TK, NU*TK);
        for (int t = 0; t < TK; ++t) {
            Rblock(2*t,   2*t)   = Rd(0);
            Rblock(2*t+1, 2*t+1) = Rd(1);
        }

        // D: (NU*(TK-1)) x (NU*TK) = 14 x 16
        int ndiff = NU*(TK-1);
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ndiff, NU*TK);
        for (int j = 0; j < TK-1; ++j) {
            D(2*j,   2*j)   = -1; D(2*j,   2*j+2) = 1;
            D(2*j+1, 2*j+1) = -1; D(2*j+1, 2*j+3) = 1;
        }
        Eigen::MatrixXd Rdblock = Eigen::MatrixXd::Zero(ndiff, ndiff);
        for (int j = 0; j < TK-1; ++j) {
            Rdblock(2*j,   2*j)   = Rdd(0);
            Rdblock(2*j+1, 2*j+1) = Rdd(1);
        }
        Eigen::MatrixXd Puv = 2.0 * (Rblock + D.transpose() * Rdblock * D);

        // Assemble full dense P then extract upper-triangular CSC
        Eigen::MatrixXd Pfull = Eigen::MatrixXd::Zero(NW, NW);
        // State part: 2*Q diagonal for t=0..TK-1, 2*Qf at t=TK
        for (int t = 0; t <= TK; ++t) {
            const Eigen::Vector4d& qd = (t < TK) ? Qd : Qfd;
            for (int r = 0; r < NXK; ++r)
                Pfull(NXK*t+r, NXK*t+r) = 2.0 * qd(r);
        }
        // Control part
        int xoff = NXK*(TK+1);  // 36
        Pfull.block(xoff, xoff, NU*TK, NU*TK) = Puv;

        // upper triangular CSC
        {
            std::vector<std::vector<std::pair<int,double>>> cols(NW);
            for (int c = 0; c < NW; ++c)
                for (int r = 0; r <= c; ++r)
                    if (std::abs(Pfull(r,c)) > 1e-12)
                        cols[c].push_back({r, Pfull(r,c)});

            P_p_.resize(NW+1, 0);
            for (int c = 0; c < NW; ++c) {
                P_p_[c+1] = P_p_[c] + (OSQPInt)cols[c].size();
                for (auto& [r, v] : cols[c]) { P_i_.push_back((OSQPInt)r); P_x_.push_back(v); }
            }
        }

        // ── A matrix (68x52) ─────────────────────────────────────────────────
        // Row layout:
        //   [0..31]  dynamics  (NXK*TK=32 rows)
        //   [32..35] init state (NXK=4)
        //   [36..44] speed     (TK+1=9)
        //   [45..52] accel     (TK=8)
        //   [53..60] steer     (TK=8)
        //   [61..67] dsteer    (TK-1=7)
        //
        // Build A as dense then convert to CSC, but also record positions
        // of the dynamics entries for fast update.

        // We build column by column.
        // Column layout of w: [x0..x8 (4 each), u0..u7 (2 each)]
        // x-col offset for time t: NXK*t
        // u-col offset for time t: NXK*(TK+1) + NU*t

        // Use Eigen sparse for convenience
        Eigen::SparseMatrix<double, Eigen::ColMajor> Asp(NCON, NW);
        using T = Eigen::Triplet<double>;
        std::vector<T> trips;

        // Dynamics rows: for step t, rows [NXK*t .. NXK*(t+1)-1]
        // xk[:,t+1] - A[t]*xk[:,t] - B[t]*uk[:,t] = C[t]
        // Column of xk[:,t]   : NXK*t
        // Column of xk[:,t+1] : NXK*(t+1)
        // Column of uk[:,t]   : NXK*(TK+1) + NU*t
        //
        // For the initial build use A[t]=0, B[t]=0 (we fill C=0 too).
        // The identity part (xk[:,t+1]) is constant.
        for (int t = 0; t < TK; ++t) {
            int row0 = NXK * t;
            int col_xnext = NXK * (t+1);
            // +I for next state
            for (int r = 0; r < NXK; ++r)
                trips.push_back({row0+r, col_xnext+r, 1.0});
            // -A[t] for current state (initially zero, will be overwritten)
            int col_x = NXK * t;
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NXK; ++c2)
                    trips.push_back({row0+r, col_x+c2, 0.0});
            // -B[t] for control (initially zero)
            int col_u = NXK*(TK+1) + NU*t;
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NU; ++c2)
                    trips.push_back({row0+r, col_u+c2, 0.0});
        }

        // Initial state rows [32..35]: I_4 on x[:,0]
        for (int r = 0; r < NXK; ++r)
            trips.push_back({ROW_INIT+r, r, 1.0});

        // Speed rows [36..44]: xk[2,t] at col NXK*t+2
        for (int t = 0; t <= TK; ++t)
            trips.push_back({ROW_SPEED+t, NXK*t+2, 1.0});

        // Accel rows [45..52]: uk[0,t] at col 36+NU*t
        for (int t = 0; t < TK; ++t)
            trips.push_back({ROW_ACCEL+t, NXK*(TK+1)+NU*t, 1.0});

        // Steer rows [53..60]: uk[1,t] at col 36+NU*t+1
        for (int t = 0; t < TK; ++t)
            trips.push_back({ROW_STEER+t, NXK*(TK+1)+NU*t+1, 1.0});

        // Dsteer rows [61..67]: uk[1,t+1]-uk[1,t]
        for (int j = 0; j < TK-1; ++j) {
            trips.push_back({ROW_DSTR+j, NXK*(TK+1)+NU*j+1,     -1.0});
            trips.push_back({ROW_DSTR+j, NXK*(TK+1)+NU*(j+1)+1,  1.0});
        }

        Asp.setFromTriplets(trips.begin(), trips.end());
        Asp.makeCompressed();

        // Copy CSC data (Eigen uses int internally; convert to OSQPInt)
        int nnz_A = Asp.nonZeros();
        A_x_.assign(Asp.valuePtr(), Asp.valuePtr() + nnz_A);
        A_i_.resize(nnz_A);
        for (int k = 0; k < nnz_A; ++k) A_i_[k] = (OSQPInt)Asp.innerIndexPtr()[k];
        A_p_.resize(NW+1);
        for (int k = 0; k <= NW; ++k) A_p_[k] = (OSQPInt)Asp.outerIndexPtr()[k];

        // ── locate dynamics entries in A_x_ ──────────────────────────────────
        // Build col -> list of (row, value_index) for quick lookup.
        std::vector<std::vector<std::pair<int,int>>> col_map(NW);
        for (int col = 0; col < NW; ++col) {
            for (int idx = (int)A_p_[col]; idx < (int)A_p_[col+1]; ++idx) {
                col_map[col].push_back({(int)A_i_[idx], idx});
            }
        }
        auto find_idx = [&](int row, int col) -> int {
            for (auto& [r,i] : col_map[col]) if (r == row) return i;
            return -1;
        };

        dyn_idx_.resize(TK);
        for (int t = 0; t < TK; ++t) {
            int row0   = NXK * t;
            int col_x  = NXK * t;
            int col_u  = NXK*(TK+1) + NU*t;
            // A entries: col_x + c, row row0+r
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NXK; ++c2)
                    dyn_idx_[t].Aidx[r*NXK+c2] = find_idx(row0+r, col_x+c2);
            // B entries
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NU; ++c2)
                    dyn_idx_[t].Bidx[r*NU+c2] = find_idx(row0+r, col_u+c2);
        }

        // ── q vector (will be filled per-solve) ──────────────────────────────
        q_vec_.assign(NW, 0.0);

        // ── l/u bounds ───────────────────────────────────────────────────────
        l_vec_.assign(NCON, 0.0);
        u_vec_.assign(NCON, 0.0);
        // dynamics: l=u=0 (updated per-solve with Ck)
        // init state: l=u=x0 (updated per-solve)
        // speed bounds
        for (int t = 0; t <= TK; ++t) {
            l_vec_[ROW_SPEED+t] = MIN_SPEED;
            u_vec_[ROW_SPEED+t] = MAX_SPEED;
        }
        // accel bounds
        for (int t = 0; t < TK; ++t) {
            l_vec_[ROW_ACCEL+t] = -MAX_ACCEL;
            u_vec_[ROW_ACCEL+t] =  MAX_ACCEL;
        }
        // steer bounds
        for (int t = 0; t < TK; ++t) {
            l_vec_[ROW_STEER+t] = MIN_STEER;
            u_vec_[ROW_STEER+t] = MAX_STEER;
        }
        // dsteer bounds
        for (int j = 0; j < TK-1; ++j) {
            l_vec_[ROW_DSTR+j] = -MAX_DSTEER * DTK;
            u_vec_[ROW_DSTR+j] =  MAX_DSTEER * DTK;
        }

        // ── OSQP setup ───────────────────────────────────────────────────────
        OSQPSettings settings;
        osqp_set_default_settings(&settings);
        settings.verbose      = 0;
        settings.warm_starting = 1;
        settings.polishing    = 0;
        settings.max_iter     = 4000;
        settings.eps_abs      = 1e-3;
        settings.eps_rel      = 1e-3;

        OSQPCscMatrix P_mat, A_mat;
        OSQPCscMatrix_set_data(&P_mat, (OSQPInt)NW, (OSQPInt)NW, (OSQPInt)P_x_.size(),
                               P_x_.data(), P_i_.data(), P_p_.data());
        OSQPCscMatrix_set_data(&A_mat, (OSQPInt)NCON, (OSQPInt)NW, (OSQPInt)A_x_.size(),
                               A_x_.data(), A_i_.data(), A_p_.data());

        OSQPInt ret = osqp_setup(&solver_, &P_mat, q_vec_.data(),
                                 &A_mat, l_vec_.data(), u_vec_.data(),
                                 (OSQPInt)NCON, (OSQPInt)NW, &settings);
        if (ret != 0)
            RCLCPP_ERROR(get_logger(), "OSQP setup failed: %d", (int)ret);

        w_prev_.assign(NW, 0.0);
        y_prev_.assign(NCON, 0.0);
    }

    // ── per-solve QP update + solve ──────────────────────────────────────────
    bool solve_mpc(const Eigen::Vector4d& x0,
                   const Eigen::Matrix<double,NXK,TK+1>& ref,
                   const Eigen::Matrix<double,NXK,TK+1>& path_pred,
                   std::vector<double>& oa_out,
                   std::vector<double>& od_out,
                   std::vector<double>& ox_out,
                   std::vector<double>& oy_out)
    {
        // 1. Update q (reference tracking cost)
        for (int t = 0; t < TK; ++t) {
            for (int r = 0; r < NXK; ++r)
                q_vec_[NXK*t+r] = -2.0 * ((t < TK) ? Qd(r) : Qfd(r)) * ref(r, t);
        }
        for (int r = 0; r < NXK; ++r)
            q_vec_[NXK*TK+r] = -2.0 * Qfd(r) * ref(r, TK);
        std::fill(q_vec_.begin()+NXK*(TK+1), q_vec_.end(), 0.0);

        // 2. Update dynamics in A and C
        for (int t = 0; t < TK; ++t) {
            double v   = path_pred(2, t);
            double phi = path_pred(3, t);
            double delta = (t < (int)od_.size()) ? od_[t] : 0.0;
            auto mm = get_model_matrix(v, phi, delta);

            // -A[t] goes into the dynamics constraint rows
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NXK; ++c2) {
                    int idx = dyn_idx_[t].Aidx[r*NXK+c2];
                    if (idx >= 0) A_x_[idx] = -mm.A(r, c2);
                }
            // -B[t]
            for (int r = 0; r < NXK; ++r)
                for (int c2 = 0; c2 < NU; ++c2) {
                    int idx = dyn_idx_[t].Bidx[r*NU+c2];
                    if (idx >= 0) A_x_[idx] = -mm.B(r, c2);
                }

            // C goes into bounds l=u for dynamics rows
            for (int r = 0; r < NXK; ++r) {
                l_vec_[NXK*t+r] = mm.C(r);
                u_vec_[NXK*t+r] = mm.C(r);
            }
        }

        // 3. Initial state constraint
        for (int r = 0; r < NXK; ++r) {
            l_vec_[ROW_INIT+r] = x0(r);
            u_vec_[ROW_INIT+r] = x0(r);
        }

        // 4. Push updates to OSQP
        osqp_update_data_vec(solver_, q_vec_.data(), l_vec_.data(), u_vec_.data());
        osqp_update_data_mat(solver_, nullptr, nullptr, 0,
                             A_x_.data(), nullptr, (OSQPInt)A_x_.size());

        // 5. Warm start
        osqp_warm_start(solver_, w_prev_.data(), y_prev_.data());

        // 6. Solve
        osqp_solve(solver_);

        auto* info = solver_->info;
        bool ok = (info->status_val == OSQP_SOLVED ||
                   info->status_val == OSQP_SOLVED_INACCURATE);

        if (!ok) {
            RCLCPP_WARN(get_logger(), "OSQP: %s", info->status);
            return false;
        }

        auto* sol = solver_->solution;
        // Save warm-start
        std::copy(sol->x, sol->x + NW, w_prev_.begin());
        std::copy(sol->y, sol->y + NCON, y_prev_.begin());

        int xoff = NXK*(TK+1);
        oa_out.resize(TK); od_out.resize(TK);
        ox_out.resize(TK+1); oy_out.resize(TK+1);
        for (int t = 0; t < TK; ++t) {
            oa_out[t] = sol->x[xoff + NU*t];
            od_out[t] = sol->x[xoff + NU*t + 1];
        }
        for (int t = 0; t <= TK; ++t) {
            ox_out[t] = sol->x[NXK*t];
            oy_out[t] = sol->x[NXK*t+1];
        }
        return true;
    }

    // ── pose callback ────────────────────────────────────────────────────────
    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        auto& pos  = msg->pose.position;
        auto& quat = msg->pose.orientation;
        double yaw = std::atan2(
            2.0*(quat.w*quat.z + quat.x*quat.y),
            1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z));
        double v = last_v_;

        Eigen::Vector4d x0{pos.x, pos.y, v, yaw};

        // reference trajectory
        Eigen::Matrix<double, NXK, TK+1> ref;
        calc_ref_trajectory(pos.x, pos.y, v, yaw, ref);

        // predicted motion
        auto path_pred = predict_motion(x0, oa_, od_);

        // solve
        std::vector<double> oa_new, od_new, ox, oy;
        bool ok = solve_mpc(x0, ref, path_pred, oa_new, od_new, ox, oy);

        if (ok) {
            oa_ = oa_new;
            od_ = od_new;
        }

        ackermann_msgs::msg::AckermannDriveStamped drive;
        drive.drive.steering_angle = float(od_[0]);
        drive.drive.speed          = float(std::clamp(v + oa_[0] * DTK,
                                                       MIN_SPEED, MAX_SPEED));
        drive_pub_->publish(drive);

        if (ok) {
            publish_predicted_traj(ox, oy);
        }
        publish_ref_traj(ref);
    }

    // ── visualisation helpers ─────────────────────────────────────────────────
    void publish_cur_waypoint_impl(double x, double y) const
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = now();
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = x; m.pose.position.y = y;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 1.0; m.color.a = 1.0;
        cur_goal_pub_->publish(m);
    }

    void publish_all_waypoints()
    {
        visualization_msgs::msg::MarkerArray arr;
        for (int i = 0; i < INTERP_N; i += 100) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp = now();
            m.id = i;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = wp_x_[i]; m.pose.position.y = wp_y_[i];
            m.pose.orientation.w = 1.0;
            m.scale.x = m.scale.y = m.scale.z = 0.15;
            m.color.g = 1.0; m.color.a = 1.0;
            arr.markers.push_back(m);
        }
        marker_pub_->publish(arr);
    }

    void publish_predicted_traj(const std::vector<double>& ox,
                                 const std::vector<double>& oy)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = now();
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.10;
        m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0;
        for (size_t i = 0; i < ox.size(); ++i) {
            geometry_msgs::msg::Point p;
            p.x = ox[i]; p.y = oy[i];
            m.points.push_back(p);
        }
        pred_pub_->publish(m);
    }

    void publish_ref_traj(const Eigen::Matrix<double,NXK,TK+1>& ref)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = now();
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.03;
        m.color.r = 1.0; m.color.g = 1.0; m.color.a = 1.0;
        for (int i = 0; i <= TK; ++i) {
            geometry_msgs::msg::Point p;
            p.x = ref(0,i); p.y = ref(1,i);
            m.points.push_back(p);
        }
        ref_pub_->publish(m);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPC>());
    rclcpp::shutdown();
    return 0;
}
