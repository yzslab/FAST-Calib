/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "mono_qr_pattern");
    ros::NodeHandle nh;

    // 读取参数
    Params params = loadParameters(nh);

    // 初始化 QR 检测
    QRDetectPtr qrDetectPtr;
    qrDetectPtr.reset(new QRDetect(nh, params));

    DataPreprocessPtr dataPreprocessPtr;
    dataPreprocessPtr.reset(new DataPreprocess(params));

    // 读取图像和点云
    cv::Mat img_input = dataPreprocessPtr->img_input_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_input = dataPreprocessPtr->cloud_input_;
    
    // 检测 QR 码
    PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
    qr_center_cloud->reserve(4);
    qrDetectPtr->detect_qr(img_input, qr_center_cloud);

    // 估计标定板的 3D 范围
    if (params.cameraextrinR.size() > 0 && params.cameraextrinT.size() > 0) {
      cv::Mat_<float> prior_lidar2camera(4, 4);
      prior_lidar2camera << params.cameraextrinR[0], params.cameraextrinR[1], params.cameraextrinR[2], params.cameraextrinT[0],
          params.cameraextrinR[3], params.cameraextrinR[4], params.cameraextrinR[5], params.cameraextrinT[1],
          params.cameraextrinR[6], params.cameraextrinR[7], params.cameraextrinR[8], params.cameraextrinT[2],
          0.f, 0.f, 0.f, 1.f;
      std::cout << "prior_lidar2camera=" << prior_lidar2camera << std::endl;

      cv::Mat prior_camera2lidar;
      cv::invert(prior_lidar2camera, prior_camera2lidar);
      std::cout << "prior_camera2lidar=" << prior_camera2lidar << std::endl;

      cv::Mat prior_camera2lidar_R = prior_camera2lidar(cv::Rect(0, 0, 3, 3));
      cv::Mat prior_camera2lidar_T;
      cv::repeat(prior_camera2lidar(cv::Rect(3, 0, 1, 3)), 1, 2, prior_camera2lidar_T);
      std::cout << "prior_camera2lidar_R=" << prior_camera2lidar_R << std::endl;
      std::cout << "prior_camera2lidar_T=" << prior_camera2lidar_T << std::endl;

      cv::Mat prior_board_bounding_box_in_lidar_space = prior_camera2lidar_R * qrDetectPtr->board_bounding_box_in_camera_space + prior_camera2lidar_T;
      std::cout << "prior_board_bounding_box_in_lidar_space=" << prior_board_bounding_box_in_lidar_space << std::endl;

      params.x_min = minf(prior_board_bounding_box_in_lidar_space.at<float>(0, 0), prior_board_bounding_box_in_lidar_space.at<float>(0, 1)) - 0.2;
      params.x_max = maxf(prior_board_bounding_box_in_lidar_space.at<float>(0, 0), prior_board_bounding_box_in_lidar_space.at<float>(0, 1)) + 0.2;
      params.y_min = minf(prior_board_bounding_box_in_lidar_space.at<float>(1, 0), prior_board_bounding_box_in_lidar_space.at<float>(1, 1)) - 0.2;
      params.y_max = maxf(prior_board_bounding_box_in_lidar_space.at<float>(1, 0), prior_board_bounding_box_in_lidar_space.at<float>(1, 1)) + 0.2;
      params.z_min = minf(prior_board_bounding_box_in_lidar_space.at<float>(2, 0), prior_board_bounding_box_in_lidar_space.at<float>(2, 1)) - 0.2;
      params.z_max = maxf(prior_board_bounding_box_in_lidar_space.at<float>(2, 0), prior_board_bounding_box_in_lidar_space.at<float>(2, 1)) + 0.2;

      std::cout << "x_min=" << params.x_min << ", x_max=" << params.x_max << std::endl;
      std::cout << "y_min=" << params.y_min << ", y_max=" << params.y_max << std::endl;
      std::cout << "z_min=" << params.z_min << ", z_max=" << params.z_max << std::endl;
    }

    // 初始化 LiDAR 检测器
    LidarDetectPtr lidarDetectPtr;
    lidarDetectPtr.reset(new LidarDetect(nh, params));


    // 检测 LiDAR 数据
    PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
    lidar_center_cloud->reserve(4);
    lidarDetectPtr->detect_lidar(cloud_input, lidar_center_cloud);

    // 对 QR 和 LiDAR 检测到的圆心进行排序
    PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
    sortPatternCenters(qr_center_cloud, qr_centers, "camera");
    sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar");

    // 计算外参
    Eigen::Matrix4f transformation;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);

    // 将 LiDAR 点云转换到 QR 码坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);
    
    double rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    if (rmse > 0) 
    {
      std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
      << rmse << " m" << RESET << std::endl;
    }

    std::cout << BOLDYELLOW << "[Result] Extrinsic parameters T_cam_lidar: " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(cloud_input, transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, colored_cloud);

    cloud_input->width = static_cast<uint32_t>(cloud_input->points.size());
    cloud_input->height = 1;
    if (pcl::io::savePCDFileASCII(params.output_path + "/cloud_input.pcd", *cloud_input) == 0) 
    {
      std::cout << BOLDYELLOW << "[Result] Saved input point cloud to: " << BOLDWHITE << params.output_path << "/cloud_input.pcd" << RESET << std::endl;
    } 

    saveCalibrationResults(params, transformation, colored_cloud, qrDetectPtr->imageCopy_);

    ros::Publisher colored_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("colored_cloud", 1);
    ros::Publisher aligned_lidar_centers_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_lidar_centers", 1);

    // 主循环
    ros::Rate rate(1);
    while (ros::ok()) 
    {
      if (DEBUG) 
      {
        // 发布 QR 检测结果
        sensor_msgs::PointCloud2 qr_centers_msg;
        pcl::toROSMsg(*qr_centers, qr_centers_msg);
        qr_centers_msg.header.stamp = ros::Time::now();
        qr_centers_msg.header.frame_id = "map";
        qrDetectPtr->qr_pub_.publish(qr_centers_msg);

        // 发布 LiDAR 检测结果
        sensor_msgs::PointCloud2 lidar_centers_msg;
        pcl::toROSMsg(*lidar_centers, lidar_centers_msg);
        lidar_centers_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_pub_.publish(lidar_centers_msg);

        // 发布中间结果
        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getFilteredCloud(), filtered_cloud_msg);
        filtered_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->filtered_pub_.publish(filtered_cloud_msg);

        sensor_msgs::PointCloud2 plane_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getPlaneCloud(), plane_cloud_msg);
        plane_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->plane_pub_.publish(plane_cloud_msg);

        sensor_msgs::PointCloud2 aligned_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getAlignedCloud(), aligned_cloud_msg);
        aligned_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->aligned_pub_.publish(aligned_cloud_msg);

        sensor_msgs::PointCloud2 edge_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getEdgeCloud(), edge_cloud_msg);
        edge_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->edge_pub_.publish(edge_cloud_msg);

        sensor_msgs::PointCloud2 lidar_centers_z0_msg;
        pcl::toROSMsg(*lidarDetectPtr->getCenterZ0Cloud(), lidar_centers_z0_msg);
        lidar_centers_z0_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_z0_pub_.publish(lidar_centers_z0_msg);

        // 发布外参变换后的LiDAR点云
        sensor_msgs::PointCloud2 aligned_lidar_centers_msg;
        pcl::toROSMsg(*aligned_lidar_centers, aligned_lidar_centers_msg);
        aligned_lidar_centers_msg.header = qr_centers_msg.header;
        aligned_lidar_centers_pub.publish(aligned_lidar_centers_msg);

        // 发布彩色点云
        sensor_msgs::PointCloud2 colored_cloud_msg;
        pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
        colored_cloud_msg.header = qr_centers_msg.header;
        colored_cloud_pub.publish(colored_cloud_msg);

        // cv::imshow("result", qrDetectPtr->imageCopy_);
      }
      // cv::waitKey(1);
      ros::spinOnce();
      rate.sleep();
    }

    return 0;
}